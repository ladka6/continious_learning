import logging
import numpy as np
import torch
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import SimpleVitNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy

num_workers = 10

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = SimpleVitNet(args, True)
        self.args = args

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        self.train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train",)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=num_workers)

        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=48, shuffle=False, num_workers=num_workers)

        train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="test", )
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=self.args["batch_size"], shuffle=True, num_workers=num_workers)

        self._train()

    def _train(self):
        self._network.to(self._device)
        optimizer = self.get_optimizer(lr=self.args["lr"])
        scheduler = self.get_scheduler(optimizer, self.args["epochs"])

        prog_bar = tqdm(range(self.args["epochs"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self._device), targets.long().to(self._device)

                optimizer.zero_grad()
                logits = self._network(inputs)["logits"]
                loss = F.cross_entropy(logits, targets)
                l1_loss = sum(p.abs().sum() for p in self._network.backbone.tosca.parameters())
                loss = loss + self.args["l1"] * l1_loss
                loss.backward()
                optimizer.step()

                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, self.test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(self.train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(self.train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)
        self._save_tosca()
        self.replace_fc()

    def after_task(self):
        self._network.backbone.reset_tosca()
        self._known_classes = self._total_classes

    def replace_fc(self):
        self._network.eval()
        self._load_tosca(self._cur_task)
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, (_, data, label) in enumerate(self.train_loader_for_protonet):
                data = data.to(self._device)
                label = label.long().to(self._device)
                embedding = self._network.backbone(data)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        class_list = np.unique(self.train_dataset.labels)
        for class_index in class_list:
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            embedding = embedding_list[data_index]
            proto = embedding.mean(0)
            self._network.fc.weight.data[class_index] = proto

    def get_optimizer(self, lr):
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self._network.parameters()), 
                                  momentum=0.9, 
                                  lr=lr,
                                  weight_decay=self.args["weight_decay"]
                                  )
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self._network.parameters()),
                                   lr=lr,
                                   weight_decay=self.args["weight_decay"]
                                   )
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self._network.parameters()),
                                    lr=lr, 
                                    weight_decay=self.args["weight_decay"]
                                    )
        return optimizer
    
    def get_scheduler(self, optimizer, epoch):
        if self.args["scheduler"] == 'constant':
            scheduler = None
        elif self.args["scheduler"] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, 
                                                             T_max=epoch, 
                                                             eta_min=1e-8)
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, 
                                                       milestones=self.args["milestones"], 
                                                       )
        return scheduler
    
    def get_lowentropy_logits(self, inputs):
        entropies = []
        logits_list = []
        for i in range(self._cur_task+1):
            self._load_tosca(i)
            outputs = self._network(inputs)["logits"]
            logits_list.append(outputs)
            probabilities = F.softmax(outputs, dim=1)
            batch_entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-9), dim=1)
            average_entropy = torch.mean(batch_entropy)
            entropies.append(average_entropy.item())
        task_idx = int(torch.argmin(torch.tensor(entropies)))

        return logits_list[task_idx]
    
    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for i, (_, inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(self._device), targets.long().to(self._device)
            with torch.no_grad():
                outputs=self.get_lowentropy_logits(inputs)
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _save_tosca(self):
        path = f"tosca/task{self._cur_task}.pth"
        tosca_state_dict = {
            name: param 
            for name, param in self._network.state_dict().items() 
            if 'tosca' in name
        }
        torch.save(tosca_state_dict, path)
        logging.info(f"tosca parameters saved to {path}.")

    def _load_tosca(self, idx):
        path = f"tosca/task{idx}.pth"
        tosca_state_dict = torch.load(path)
        current_state_dict = self._network.state_dict()
        current_state_dict.update(tosca_state_dict)
        self._network.load_state_dict(current_state_dict)
        

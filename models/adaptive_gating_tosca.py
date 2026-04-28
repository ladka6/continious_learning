import os
import logging
import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils.inc_net import SimpleAdaptiveVitNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
from utils.gating import FeatureStatsCollector, TaskGate, generate_samples

num_workers = 4
TOSCA_DIR = "tosca"


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = SimpleAdaptiveVitNet(args, True)
        self.args = args
        
        self.task_ranges = []
        
        self._routing_mode = args.get("routing_mode", args.get("task_selector", "entropy"))
        
        self._use_gate = self._routing_mode == "gate"
        
        self._gate = None
        self._task_feature_stats = {}
        
        os.makedirs(TOSCA_DIR, exist_ok=True)

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        self.train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train", mode="train")
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.args["batch_size"],
            shuffle=True, num_workers=num_workers)

        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(
            test_dataset, batch_size=48, shuffle=False, num_workers=num_workers)

        train_dataset_for_protonet = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train", mode="test")
        self.train_loader_for_protonet = DataLoader(
            train_dataset_for_protonet, batch_size=self.args["batch_size"],
            shuffle=True, num_workers=num_workers)

        self._train()

    def _train(self):
        self._network.to(self._device)

        # Trainable params: adapters (task 0 only) + tosca (always)
        if self._cur_task == 0:
            self._set_trainable(adapters=True, tosca=True, w_rand=True)
        else:
            self._set_trainable(adapters=False, tosca=True, w_rand=False)
        
        # self._init_prototypes()

        total = sum(p.numel() for p in self._network.parameters())
        trainable = sum(p.numel() for p in self._network.parameters() if p.requires_grad)
        logging.info(f"Task {self._cur_task}: {trainable:,}/{total:,} trainable")

        optimizer = self._get_optimizer(self.args["lr"])
        scheduler = self._get_scheduler(optimizer, self.args["epochs"])

        prog_bar = tqdm(range(self.args["epochs"]))
        for epoch in prog_bar:
            self._network.train()
            losses, correct, total_n = 0.0, 0, 0
            for _, inputs, targets in self.train_loader:
                inputs = inputs.to(self._device)
                targets = targets.long().to(self._device)

                optimizer.zero_grad()
                logits = self._network(inputs)["logits"]
                ce = F.cross_entropy(logits, targets)
                l1 = sum(p.abs().sum() for p in self._network.backbone.tosca.parameters())
                loss = ce + self.args["l1"] * l1
                loss.backward()
                optimizer.step()

                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets).cpu().sum()
                total_n += len(targets)

            if scheduler is not None:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total_n, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, self.test_loader)
                info = f"Task {self._cur_task} E{epoch+1}/{self.args['epochs']} => Loss {losses/len(self.train_loader):.3f} Train {train_acc:.2f} Test {test_acc:.2f}"
            else:
                info = f"Task {self._cur_task} E{epoch+1}/{self.args['epochs']} => Loss {losses/len(self.train_loader):.3f} Train {train_acc:.2f}"
            prog_bar.set_description(info)
        logging.info(info)
        
        # After task 0, lock adapters forever
        if self._cur_task == 0:
            self._network.backbone.freeze_adapters()
            self._network.backbone.freeze_w_rand()
        
        self._save_tosca()
        
        
        if self._use_gate:
            self._collect_current_task_stats()
            self._train_gate()

        self._replace_fc()
    
    def _get_backbone(self):
        if isinstance(self._network, torch.nn.DataParallel):
            return self._network.module.backbone
        return self._network.backbone
        
    def _extract_backbone_features(self, inputs):
        backbone = self._get_backbone()
        return backbone.forward_features(inputs)

    def _prepare_gate_features(self, features):
        if self.args.get("gate_normalize_features", True):
            return F.normalize(features, p=2, dim=1)
        return features

    def _init_or_extend_gate(self):
        num_tasks = self._cur_task + 1
        if self._gate is None:
            self._gate = TaskGate(
                input_dim=self._network.backbone.embed_dim,
                num_tasks=num_tasks,
                hidden_dim=int(self.args.get("gate_hidden_dim", 0)),
            ).to(self._device)
        else:
            self._gate.extend(num_tasks)
            self._gate.to(self._device)
            
    def _collect_current_task_stats(self):
        self._network.eval()
        collector = FeatureStatsCollector(feature_dim=self._network.backbone.embed_dim)
        with torch.no_grad():
            for _, data, label in self.train_loader_for_protonet:
                data = data.to(self._device)
                label = label.long().to(self._device)
                features = self._extract_backbone_features(data)
                features = self._prepare_gate_features(features)
                collector.update(features, label)
                
        class_stats = collector.compute_mean_variance()
        self._task_feature_stats[self._cur_task] = class_stats
        
    def _collect_all_synthetic_features(self):
        synthetic_per_class = int(self.args.get("gate_synthetic_per_class", 200))
        min_variance = self.args.get("gate_min_variance", 1e-6)

        all_features, all_targets = [], []
        for task_idx in range(self._cur_task + 1):
            class_stats = self._task_feature_stats.get(task_idx, {})
            sampled_features, _ = generate_samples(
                class_stats,
                n_samples=synthetic_per_class,
                min_variance=min_variance,
                device="cpu",
            )
            if sampled_features.numel() == 0:
                continue
            task_targets = torch.full(
                (sampled_features.size(0),), task_idx, dtype=torch.long
            )
            all_features.append(sampled_features)
            all_targets.append(task_targets)

        if len(all_features) == 0:
            return (
                torch.empty(0, self._network.backbone.embed_dim),
                torch.empty(0, dtype=torch.long),
            )

        return torch.cat(all_features, dim=0), torch.cat(all_targets, dim=0)
        
    def _train_gate(self):
        self._init_or_extend_gate()

        train_x, train_y = self._collect_all_synthetic_features()

        if train_x.numel() == 0:
            logging.warning("No synthetic features available for gate training.")
            return

        gate_batch_size = int(self.args.get("gate_batch_size", 256))
        gate_epochs = int(self.args.get("gate_epochs", 10))
        gate_lr = float(self.args.get("gate_lr", 1e-3))
        gate_wd = float(self.args.get("gate_weight_decay", 0.0))

        gate_dataset = TensorDataset(train_x, train_y)
        gate_loader = DataLoader(
            gate_dataset,
            batch_size=gate_batch_size,
            shuffle=True,
            num_workers=0,
        )

        optimizer = optim.Adam(self._gate.parameters(), lr=gate_lr, weight_decay=gate_wd)
        criterion = torch.nn.CrossEntropyLoss()

        self._gate.train()
        for epoch in range(gate_epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for features, task_ids in gate_loader:
                features = features.to(self._device)
                task_ids = task_ids.to(self._device)

                optimizer.zero_grad()
                logits = self._gate(features)
                loss = criterion(logits, task_ids)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct += (preds == task_ids).sum().item()
                total += task_ids.size(0)


        self._gate.eval()
        self._save_gate()

    def _save_gate(self):
        if self._gate is None:
            return

        path = f"tosca/gate_task{self._cur_task}.pth"
        torch.save(
            {
                "state_dict": self._gate.state_dict(),
                "num_tasks": self._cur_task + 1,
                "input_dim": self._network.backbone.embed_dim,
                "hidden_dim": int(self.args.get("gate_hidden_dim", 0)),
            },
            path,
        )
        logging.info(f"Gate parameters saved to {path}.")


    def after_task(self):
        self._network.backbone.reset_tosca()
        self._known_classes = self._total_classes

    def _set_trainable(self, adapters, tosca, w_rand=False):
        for p in self._network.parameters():
            p.requires_grad = False
        if adapters:
            for name, p in self._network.backbone.vit.named_parameters():
                if 'adaptmlp' in name:
                    p.requires_grad = True
        if tosca:
            for p in self._network.backbone.tosca.parameters():
                p.requires_grad = True
        if w_rand:
            self._network.backbone.W_rand.requires_grad = True
        
    def _replace_fc(self):
        """NCM prototypes in M-dim space using the just-trained tosca."""
        self._network.eval()
        self._load_tosca(self._cur_task)
        embeddings, labels = [], []
        with torch.no_grad():
            for _, data, label in self.train_loader_for_protonet:
                data = data.to(self._device)
                label = label.long().to(self._device)
                emb = self._network.backbone(data)  # [B, M]
                embeddings.append(emb.cpu())
                labels.append(label.cpu())
        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.cat(labels, dim=0)
        for c in np.unique(self.train_dataset.labels):
            idx = (labels == c).nonzero().squeeze(-1)
            self._network.fc.weight.data[c] = embeddings[idx].mean(0)

    def _save_tosca(self):
        path = os.path.join(TOSCA_DIR, f"task{self._cur_task}.pth")
        state = {n: p for n, p in self._network.state_dict().items() if 'tosca' in n}
        torch.save(state, path)
        logging.info(f"saved tosca for task {self._cur_task} to {path}")

    def _load_tosca(self, idx):
        path = os.path.join(TOSCA_DIR, f"task{idx}.pth")
        state = torch.load(path, map_location=self._device)
        cur = self._network.state_dict()
        cur.update(state)
        self._network.load_state_dict(cur)

    def _get_lowentropy_logits(self, inputs):
        all_logits = []     # one [B, total_classes] per task
        all_entropies = []  # one [B] per task

        for t in range(self._cur_task + 1):
            self._load_tosca(t)
            outputs = self._network(inputs)["logits"]  # [B, total_classes]
            all_logits.append(outputs)
            probs = F.softmax(outputs, dim=1)
            ent = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)  # [B], per-sample
            all_entropies.append(ent)

        # Stack: [num_tasks, B]
        entropies = torch.stack(all_entropies, dim=0)
        logits = torch.stack(all_logits, dim=0)  # [num_tasks, B, total_classes]

        # Per-sample argmin over tasks: [B]
        chosen_task = torch.argmin(entropies, dim=0)

        # Gather the chosen task's logits for each sample
        B = inputs.size(0)
        selected = logits[chosen_task, torch.arange(B, device=inputs.device)]  # [B, total_classes]

        return selected

    def _get_gate_routed_logits(self, inputs):
        features = self._extract_backbone_features(inputs)
        features = self._prepare_gate_features(features)

        self._gate.eval()
        gate_logits = self._gate(features)
        chosen_task = torch.argmax(gate_logits, dim=1)  # [B]

        B = inputs.size(0)
        total_classes = self._network.fc.out_features
        out_logits = torch.zeros(B, total_classes, device=inputs.device)

        for t in chosen_task.unique().tolist():
            mask = chosen_task == t
            self._load_tosca(t)
            out_logits[mask] = self._network(inputs[mask])["logits"]

        return out_logits

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, inputs, targets in loader:
            inputs = inputs.to(self._device)
            targets = targets.long().to(self._device)
            with torch.no_grad():
                if self._use_gate:
                    outputs = self._get_gate_routed_logits(inputs)
                else:
                    outputs = self._get_lowentropy_logits(inputs)
            preds = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]
            y_pred.append(preds.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)

    def _get_optimizer(self, lr):
        params = filter(lambda p: p.requires_grad, self._network.parameters())
        opt = self.args["optimizer"]
        wd = self.args["weight_decay"]
        if opt == "sgd":
            return optim.SGD(params, momentum=0.9, lr=lr, weight_decay=wd)
        if opt == "adam":
            return optim.Adam(params, lr=lr, weight_decay=wd)
        if opt == "adamw":
            return optim.AdamW(params, lr=lr, weight_decay=wd)
        raise ValueError(f"unknown optimizer {opt}")

    def _get_scheduler(self, optimizer, epochs):
        sch = self.args["scheduler"]
        if sch == "constant":
            return None
        if sch == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-8)
        if sch == "steplr":
            return optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args["milestones"])
        raise ValueError(f"unknown scheduler {sch}")
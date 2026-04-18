import logging
import numpy as np
import torch
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from utils.gating import FeatureStatsCollector, TaskGate, generate_samples
from utils.inc_net import SimpleVitNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy

num_workers = 10


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = SimpleVitNet(args, True)
        self.args = args

        self._task_ranges = []

        self._routing_mode = args.get(
            "routing_mode", args.get("task_selector", "entropy")
        ).lower()
        self._use_gate = self._routing_mode == "gate"

        self._gate = None
        self._task_feature_stats = {}

        # RanPAC state
        self._total_classnum = None   
        self.W_rand = None            
        self.G = None                 
        self.Q = None                 
        self._ranpac_Wo = None        

    # ─────────────────────────────────────────────────────────────
    # RanPAC
    # ─────────────────────────────────────────────────────────────

    def setup_RP(self):
        """
        Called once after task 0 training completes.
        W_rand is sampled from N(0,1) and never updated again.
        G and Q live on CPU to avoid 400MB GPU allocation for M=10000.
        """
        M = self.args.get("ranpac_M", 10000)
        L = self._network.feature_dim 
        K = self._total_classnum

        self.W_rand = torch.randn(L, M)  
        self.G = torch.zeros(M, M)       
        self.Q = torch.zeros(M, K)       

        logging.info(f"RanPAC: W_rand ({L}×{M}), G ({M}×{M}), Q ({M}×{K}) initialised.")

    def optimise_ridge_parameter(self, Features, Y):
        ridges = 10.0 ** np.arange(-8, 9)
        num_val = int(Features.shape[0] * 0.8)
        losses = []
        Q_val = Features[:num_val].T @ Y[:num_val]
        G_val = Features[:num_val].T @ Features[:num_val]
        for ridge in ridges:
            Wo = torch.linalg.solve(
                G_val + ridge * torch.eye(G_val.size(0)), Q_val
            ).T
            losses.append(F.mse_loss(Features[num_val:] @ Wo.T, Y[num_val:]).item())
        best = ridges[np.argmin(losses)]
        logging.info(f"RanPAC optimal λ = {best}")
        return best

    def replace_fc(self):
        self._network.eval()
        self._load_tosca(self._cur_task)

        embedding_list = []
        label_list = []
        with torch.no_grad():
            for _, data, label in self.train_loader_for_protonet:
                data = data.to(self._device)
                label = label.long().to(self._device)
                embedding = self._network.backbone(data)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())

        Features_f = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)     

        Y = target2onehot(label_list, self._total_classnum)

        Features_h = F.relu(Features_f @ self.W_rand)

        self.G = self.G + Features_h.T @ Features_h  
        self.Q = self.Q + Features_h.T @ Y           

        # solve: Wₒ = (G + λI)⁻¹ Q  →  shape (M, K)
        ridge = self.optimise_ridge_parameter(Features_h, Y)
        self._ranpac_Wo = torch.linalg.solve(
            self.G + ridge * torch.eye(self.G.size(0)),
            self.Q
        ).to(self._device)  # (M, K) on device for fast inference

        logging.info(f"RanPAC: Wₒ updated → shape {self._ranpac_Wo.shape}")

    # ─────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        # total_classnum is fixed for the whole experiment — grab it once
        if self._total_classnum is None:
            self._total_classnum = data_manager.get_total_classnum()
            logging.info(f"Total classes in experiment: {self._total_classnum}")

        self._network.update_fc(self._total_classes)

        self._task_ranges.append((self._known_classes, self._total_classes))
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
        logging.info(f"Task ranges so far: {self._task_ranges}")

        self.train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args["batch_size"],
            shuffle=True,
            num_workers=num_workers,
        )

        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=48, shuffle=False, num_workers=num_workers
        )

        train_dataset_for_protonet = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="test",
        )
        self.train_loader_for_protonet = DataLoader(
            train_dataset_for_protonet,
            batch_size=self.args["batch_size"],
            shuffle=True,
            num_workers=num_workers,
        )

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
            for _, inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self._device), targets.long().to(
                    self._device
                )
                optimizer.zero_grad()
                logits = self._network(inputs)["logits"]
                loss = F.cross_entropy(logits, targets)
                l1_loss = sum(
                    p.abs().sum() for p in self._network.backbone.tosca.parameters()
                )
                loss = loss + self.args["l1"] * l1_loss
                loss.backward()
                optimizer.step()

                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler is not None:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            current_epoch = epoch + 1
            test_acc = self._compute_accuracy(self._network, self.test_loader)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                current_epoch,
                self.args["epochs"],
                losses / len(self.train_loader),
                train_acc,
                test_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)
        self._save_tosca()

        if self._use_gate:
            self._collect_current_task_statistics()
            self._train_gate()

        # RanPAC: init on task 0, accumulate + solve every task
        if self._cur_task == 0:
            self.setup_RP()
        self.replace_fc()

    # ─────────────────────────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────────────────────────

    def get_routed_logits(self, inputs, selector=None):
        """
        Gate selects which TOSCA checkpoint to load, then RanPAC scores the batch.
        Falls back to FC logits only if replace_fc hasn't run yet (should never
        happen outside the first epoch of task 0).
        """
        task_idx = self._get_selected_task_for_batch(inputs, selector=selector)
        self._load_tosca(task_idx)

        if self._ranpac_Wo is None:
            # safety fallback — FC path
            return self._network(inputs)["logits"]

        with torch.no_grad():
            f = self._network.backbone(inputs)   # (N, L) post-TOSCA
            h = F.relu(f @ self.W_rand.to(self._device))  # (N, M)
            scores = h @ self._ranpac_Wo          # (N, K)
        return scores

    # ─────────────────────────────────────────────────────────────
    # Gate
    # ─────────────────────────────────────────────────────────────

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

    def _collect_current_task_statistics(self):
        self._network.eval()
        collector = FeatureStatsCollector(
            feature_dim=self._network.feature_dim,
            min_variance=self.args.get("gate_min_variance", 1e-6),
        )
        with torch.no_grad():
            for _, data, label in self.train_loader_for_protonet:
                data = data.to(self._device)
                features = self._extract_backbone_features(data)
                features = self._prepare_gate_features(features)
                collector.update(features, label)

        self._task_feature_stats[self._cur_task] = collector.compute_mean_variance()

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
                torch.empty(0, self._network.feature_dim),
                torch.empty(0, dtype=torch.long),
            )

        return torch.cat(all_features, dim=0), torch.cat(all_targets, dim=0)

    def _init_or_extend_gate(self):
        num_tasks = self._cur_task + 1
        if self._gate is None:
            self._gate = TaskGate(
                input_dim=self._network.feature_dim,
                num_tasks=num_tasks,
                hidden_dim=int(self.args.get("gate_hidden_dim", 0)),
            ).to(self._device)
        else:
            self._gate.extend(num_tasks)
            self._gate.to(self._device)

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

        optimizer = optim.Adam(
            self._gate.parameters(), lr=gate_lr, weight_decay=gate_wd
        )
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
                "input_dim": self._network.feature_dim,
                "hidden_dim": int(self.args.get("gate_hidden_dim", 0)),
            },
            path,
        )
        logging.info(f"Gate parameters saved to {path}.")

    def _get_selected_task_for_batch_entropy(self, inputs):
        entropies = []
        for task_idx in range(self._cur_task + 1):
            self._load_tosca(task_idx)
            outputs = self._network(inputs)["logits"]
            probabilities = F.softmax(outputs, dim=1)
            batch_entropy = -torch.sum(
                probabilities * torch.log(probabilities + 1e-9), dim=1
            )
            entropies.append(torch.mean(batch_entropy).item())
        return int(np.argmin(entropies))

    def _get_selected_task_for_batch_gate(self, inputs):
        if self._gate is None or self._gate.num_tasks < (self._cur_task + 1):
            return self._get_selected_task_for_batch_entropy(inputs)
        self._gate.eval()
        features = self._extract_backbone_features(inputs)
        features = self._prepare_gate_features(features)
        logits = self._gate(features)
        probs = F.softmax(logits, dim=1).mean(dim=0)
        return int(torch.argmax(probs).item())

    def _get_selected_task_for_batch(self, inputs, selector=None):
        selector = (selector or self._routing_mode).lower()
        if selector == "gate":
            return self._get_selected_task_for_batch_gate(inputs)
        return self._get_selected_task_for_batch_entropy(inputs)

    # ─────────────────────────────────────────────────────────────
    # Bookkeeping
    # ─────────────────────────────────────────────────────────────

    def after_task(self):
        self._network.backbone.reset_tosca()
        self._known_classes = self._total_classes

    def eval_routing_comparison(self):
        if not self._use_gate:
            return None
        comparisons = {}
        for selector in ("entropy", "gate"):
            y_pred, y_true = self._eval_cnn_for_selector(self.test_loader, selector)
            comparisons[selector] = self._evaluate(y_pred, y_true)
        return comparisons

    def _eval_cnn_for_selector(self, loader, selector):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(self._device), targets.long().to(self._device)
            with torch.no_grad():
                outputs = self.get_routed_logits(inputs, selector=selector)
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)

    def _eval_cnn(self, loader):
        selector = self._routing_mode if self._use_gate else "entropy"
        return self._eval_cnn_for_selector(loader, selector)

    def _save_tosca(self):
        path = f"tosca/task{self._cur_task}.pth"
        tosca_state_dict = {
            name: param
            for name, param in self._network.state_dict().items()
            if "tosca" in name
        }
        torch.save(tosca_state_dict, path)
        logging.info(f"tosca parameters saved to {path}.")

    def _load_tosca(self, idx):
        path = f"tosca/task{idx}.pth"
        tosca_state_dict = torch.load(path, map_location=self._device)
        current_state_dict = self._network.state_dict()
        current_state_dict.update(tosca_state_dict)
        self._network.load_state_dict(current_state_dict)

    def get_optimizer(self, lr):
        if self.args["optimizer"] == "sgd":
            return optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                momentum=0.9,
                lr=lr,
                weight_decay=self.args["weight_decay"],
            )
        elif self.args["optimizer"] == "adam":
            return optim.Adam(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=lr,
                weight_decay=self.args["weight_decay"],
            )
        elif self.args["optimizer"] == "adamw":
            return optim.AdamW(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=lr,
                weight_decay=self.args["weight_decay"],
            )
        raise NotImplementedError("Unknown optimizer {}".format(self.args["optimizer"]))

    def get_scheduler(self, optimizer, epoch):
        if self.args["scheduler"] == "constant":
            return None
        elif self.args["scheduler"] == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=epoch, eta_min=1e-8
            )
        elif self.args["scheduler"] == "steplr":
            return optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=self.args["milestones"],
            )
        raise NotImplementedError("Unknown scheduler {}".format(self.args["scheduler"]))
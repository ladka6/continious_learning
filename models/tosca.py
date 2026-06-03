import logging
import numpy as np
import torch
import time
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from utils.gating import (
    FeatureStatsCollector,
    RidgeTaskGate,
    TaskGate,
    generate_samples,
)
from utils.inc_net import SimpleAdaptiveVitNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy

num_workers = 10


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = SimpleAdaptiveVitNet(args, True)
        self.args = args

        # Task boundaries: task_ranges[i] = (start_class, end_class)
        self._task_ranges = []

        # Gate components
        self._gate = None
        self._task_feature_stats = {}
        self._latest_task_timing = {}
        self._gate_classifier = self.args.get("gate_classifier", "ridge").lower()

    @property
    def _projection_dim(self):
        return int(self._get_backbone().M)

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)

        # Record task boundary
        self._task_ranges.append((self._known_classes, self._total_classes))
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )
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
            test_dataset, batch_size=48, shuffle=True, num_workers=num_workers
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
        task_train_start = time.perf_counter()
        self._network.to(self._device)
        self._set_trainable()
        optimizer = self.get_optimizer(lr=self.args["lr"])
        scheduler = self.get_scheduler(optimizer, self.args["epochs"])

        backbone_train_start = time.perf_counter()
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

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(self.train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)

        logging.info(info)
        backbone_train_seconds = time.perf_counter() - backbone_train_start
        self._save_tosca()

        stats_start = time.perf_counter()
        self._collect_current_task_statistics()
        stats_seconds = time.perf_counter() - stats_start
        gate_train_start = time.perf_counter()
        self._train_gate()
        gate_train_seconds = time.perf_counter() - gate_train_start

        replace_fc_start = time.perf_counter()
        self.replace_fc()
        replace_fc_seconds = time.perf_counter() - replace_fc_start
        task_train_seconds = time.perf_counter() - task_train_start

        self._latest_task_timing = {
            "backbone_train_seconds": backbone_train_seconds,
            "stats_collection_seconds": stats_seconds,
            "gate_train_seconds": gate_train_seconds,
            "replace_fc_seconds": replace_fc_seconds,
            "task_train_seconds": task_train_seconds,
        }
        logging.info(
            "Task {} timing => backbone_train {:.2f}s, stats {:.2f}s, gate_train {:.2f}s, replace_fc {:.2f}s, total_train {:.2f}s".format(
                self._cur_task,
                backbone_train_seconds,
                stats_seconds,
                gate_train_seconds,
                replace_fc_seconds,
                task_train_seconds,
            )
        )

    def _get_backbone(self):
        if isinstance(self._network, torch.nn.DataParallel):
            return self._network.module.backbone
        return self._network.backbone

    def _set_trainable(self):
        # Adapters trainable on task 0 only, frozen afterwards.
        # W_rand always frozen. Tosca + fc trainable every task.
        for p in self._network.parameters():
            p.requires_grad = False
        backbone = self._get_backbone()
        for p in backbone.tosca.parameters():
            p.requires_grad = True
        for p in self._network.fc.parameters():
            p.requires_grad = True
        if self._cur_task == 0:
            for name, p in backbone.vit.named_parameters():
                if "adaptmlp" in name:
                    p.requires_grad = True

    def _extract_backbone_features(self, inputs):
        backbone = self._get_backbone()
        return backbone.forward_features(inputs)

    def _prepare_gate_features(self, features):
        if self.args.get("gate_normalize_features", True):
            features = F.normalize(features, p=2, dim=1)
        return features

    def _project_features(self, features):
        backbone = self._get_backbone()
        return torch.relu(features @ backbone.W_rand)

    def _collect_current_task_statistics(self):
        self._network.eval()
        collector = FeatureStatsCollector(
            feature_dim=self._projection_dim,
            min_variance=self.args.get("gate_min_variance", 1e-6),
            stats_mode=self.args.get("gate_stats_mode", "diag"),
        )

        with torch.no_grad():
            for _, data, label in self.train_loader_for_protonet:
                data = data.to(self._device)
                features = self._extract_backbone_features(data)
                features = self._prepare_gate_features(features)
                features = self._project_features(features)
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
                torch.empty(0, self._projection_dim),
                torch.empty(0, dtype=torch.long),
            )

        return torch.cat(all_features, dim=0), torch.cat(all_targets, dim=0)

    def _init_or_extend_gate(self):
        num_tasks = self._cur_task + 1
        if self._gate is None:
            if self._gate_classifier == "ridge":
                self._gate = RidgeTaskGate(
                    input_dim=self._projection_dim,
                    num_tasks=num_tasks,
                    ridge_lambda=float(self.args.get("gate_ridge_lambda", 1.0)),
                ).to(self._device)
            else:
                self._gate = TaskGate(
                    input_dim=self._projection_dim,
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

        assert self._gate is not None
        if self._gate.num_tasks < 2:
            logging.info("Gate Task 0: single task, skipping training (nothing to discriminate).")
            self._gate.eval()
            self._save_gate()
            return

        if self._gate_classifier == "ridge":
            train_x = train_x.to(self._device)
            train_y = train_y.to(self._device)
            self._gate.fit(train_x, train_y)
            with torch.no_grad():
                logits = self._gate(train_x)
                loss = F.cross_entropy(logits, train_y)
                preds = torch.argmax(logits, dim=1)
                train_acc = 100.0 * (preds == train_y).sum().item() / max(
                    train_y.size(0), 1
                )
            logging.info(
                f"Gate Task {self._cur_task}, Ridge fit => Loss {loss.item():.4f}, Train_accy {train_acc:.2f}"
            )
        else:
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

                avg_loss = epoch_loss / max(len(gate_loader), 1)
                train_acc = 100.0 * correct / total if total > 0 else 0.0
                logging.info(
                    f"Gate Task {self._cur_task}, Epoch {epoch + 1}/{gate_epochs} => "
                    f"Loss {avg_loss:.4f}, Train_accy {train_acc:.2f}"
                )
                if avg_loss < 1e-4:
                    logging.info(f"Gate converged at epoch {epoch + 1}, stopping early.")
                    break

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
                "input_dim": self._projection_dim,
                "gate_classifier": self._gate_classifier,
                "gate_ridge_lambda": float(self.args.get("gate_ridge_lambda", 1.0)),
                "hidden_dim": int(self.args.get("gate_hidden_dim", 0)),
            },
            path,
        )
        logging.info(f"Gate parameters saved to {path}.")

    def _get_gate_routed_logits(self, inputs):
        if self._gate is None or self._gate.num_tasks < (self._cur_task + 1):
            raise RuntimeError(
                "Gate is not initialized or out of sync with current task count."
            )

        self._gate.eval()
        features = self._extract_backbone_features(inputs)
        features = self._prepare_gate_features(features)
        features = self._project_features(features)
        gate_logits = self._gate(features)
        chosen_task = torch.argmax(gate_logits, dim=1)

        batch_size = inputs.size(0)
        total_classes = self._network.fc.out_features
        out_logits = torch.zeros(batch_size, total_classes, device=inputs.device)

        for t in chosen_task.unique().tolist():
            mask = chosen_task == t
            self._load_tosca(int(t))
            out_logits[mask] = self._network(inputs[mask])["logits"]

        return out_logits

    def after_task(self):
        self._network.backbone.reset_tosca()
        self._known_classes = self._total_classes

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
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        class_list = np.unique(self.train_dataset.labels)
        for class_index in class_list:
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            embedding = embedding_list[data_index]
            proto = embedding.mean(0)
            self._network.fc.weight.data[class_index] = proto

    def get_optimizer(self, lr):
        if self.args["optimizer"] == "sgd":
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                momentum=0.9,
                lr=lr,
                weight_decay=self.args["weight_decay"],
            )
        elif self.args["optimizer"] == "adam":
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=lr,
                weight_decay=self.args["weight_decay"],
            )
        elif self.args["optimizer"] == "adamw":
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=lr,
                weight_decay=self.args["weight_decay"],
            )
        else:
            raise NotImplementedError(
                "Unknown optimizer {}".format(self.args["optimizer"])
            )
        return optimizer

    def get_scheduler(self, optimizer, epoch):
        if self.args["scheduler"] == "constant":
            scheduler = None
        elif self.args["scheduler"] == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=epoch, eta_min=1e-8
            )
        elif self.args["scheduler"] == "steplr":
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=self.args["milestones"],
            )
        else:
            raise NotImplementedError(
                "Unknown scheduler {}".format(self.args["scheduler"])
            )
        return scheduler

    def _eval_gate_routing(self, loader):
        if self._gate is None:
            return None

        self._gate.eval()
        correct = 0
        total = 0
        per_task = {
            task_idx: {"correct": 0, "total": 0, "predicted": 0}
            for task_idx in range(len(self._task_ranges))
        }

        with torch.no_grad():
            for _, inputs, targets in loader:
                inputs = inputs.to(self._device)
                targets = targets.long()

                features = self._extract_backbone_features(inputs)
                features = self._prepare_gate_features(features)
                features = self._project_features(features)
                gate_logits = self._gate(features)
                chosen_task = torch.argmax(gate_logits, dim=1).cpu()

                true_task = torch.zeros_like(targets)
                for t, (start, end) in enumerate(self._task_ranges):
                    mask = (targets >= start) & (targets < end)
                    true_task[mask] = t

                correct += (chosen_task == true_task).sum().item()
                total += targets.size(0)
                for task_idx in range(len(self._task_ranges)):
                    true_mask = true_task == task_idx
                    pred_mask = chosen_task == task_idx
                    per_task[task_idx]["total"] += true_mask.sum().item()
                    per_task[task_idx]["correct"] += (
                        (chosen_task[true_mask] == true_task[true_mask]).sum().item()
                    )
                    per_task[task_idx]["predicted"] += pred_mask.sum().item()

        gate_acc = 100.0 * correct / total if total > 0 else 0.0
        for task_idx, stats in per_task.items():
            task_total = stats["total"]
            stats["accuracy"] = (
                100.0 * stats["correct"] / task_total if task_total > 0 else 0.0
            )
        return {"top1": gate_acc, "per_task": per_task}

    def _compute_gate_routing_flops(self, batch_size=1):
        input_dim = int(self._network.feature_dim)
        projection_dim = int(self._projection_dim)
        if self._gate_classifier == "ridge":
            per_sample_flops = 2 * input_dim * projection_dim
            per_sample_flops += projection_dim
            per_sample_flops += 2 * projection_dim * max(self._cur_task + 1, 1)
            return {
                "per_sample": int(per_sample_flops),
                "per_batch": int(per_sample_flops * batch_size),
                "batch_size": int(batch_size),
                "num_tasks": int(max(self._cur_task + 1, 1)),
            }

        hidden_dim = int(self.args.get("gate_hidden_dim", 0))
        num_tasks = max(self._cur_task + 1, 1)

        projection_flops = 2 * input_dim * projection_dim
        relu_projection_flops = projection_dim

        if hidden_dim > 0:
            hidden_flops = 2 * projection_dim * hidden_dim + hidden_dim
            classifier_flops = 2 * hidden_dim * num_tasks
        else:
            hidden_flops = 0
            classifier_flops = 2 * projection_dim * num_tasks

        per_sample_flops = (
            projection_flops
            + relu_projection_flops
            + hidden_flops
            + classifier_flops
        )
        return {
            "per_sample": int(per_sample_flops),
            "per_batch": int(per_sample_flops * batch_size),
            "batch_size": int(batch_size),
            "num_tasks": int(num_tasks),
        }

    def eval_task(self):
        eval_start = time.perf_counter()
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)
        nme_accy = None
        gate_accy = self._eval_gate_routing(self.test_loader)
        eval_seconds = time.perf_counter() - eval_start
        num_samples = len(self.test_loader.dataset)
        gate_flops = self._compute_gate_routing_flops(
            batch_size=self.test_loader.batch_size
        )
        if gate_accy is not None:
            gate_accy["eval_seconds"] = eval_seconds
            gate_accy["ms_per_sample"] = 1000.0 * eval_seconds / max(num_samples, 1)
            gate_accy["routing_flops"] = gate_flops
        return cnn_accy, nme_accy, gate_accy

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []

        for _, (_, inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(self._device), targets.long().to(self._device)
            with torch.no_grad():
                outputs = self._get_gate_routed_logits(inputs)
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[
                1
            ]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)

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

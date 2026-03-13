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
from utils.toolkit import tensor2numpy

num_workers = 10


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = SimpleVitNet(args, True)
        self.args = args

        # Task boundaries: task_ranges[i] = (start_class, end_class)
        self._task_ranges = []
        # Task-specific test loaders for head-selection reports
        self._task_test_loaders = []
        # Selection history per task
        self._tosca_selection_history = []

        # Routing strategy: "entropy" (baseline) or "gate"
        self._routing_mode = args.get(
            "routing_mode", args.get("task_selector", "entropy")
        ).lower()
        self._use_gate = self._routing_mode == "gate"

        # Gate components
        self._gate = None
        self._task_feature_stats = {}

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
            test_dataset, batch_size=48, shuffle=False, num_workers=num_workers
        )

        task_test_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="test",
            mode="test",
        )
        task_test_loader = DataLoader(
            task_test_dataset, batch_size=48, shuffle=False, num_workers=num_workers
        )
        self._task_test_loaders.append(task_test_loader)

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

        report_batches = self.args.get("selection_report_batches", 20)

        # Baseline analysis with entropy routing
        entropy_report = self._report_tosca_module_selection(
            max_batches_per_task=report_batches,
            verbose=True,
            selector="entropy",
        )

        gate_report = None
        if self._use_gate:
            self._collect_current_task_statistics()
            self._train_gate()
            gate_report = self._report_tosca_module_selection(
                max_batches_per_task=report_batches,
                verbose=True,
                selector="gate",
            )

        self._tosca_selection_history.append(
            {
                "task": self._cur_task,
                "entropy": entropy_report,
                "gate": gate_report,
            }
        )

        self.replace_fc()

    def _get_backbone(self):
        if isinstance(self._network, torch.nn.DataParallel):
            return self._network.module.backbone
        return self._network.backbone

    def _extract_backbone_features(self, inputs):
        backbone = self._get_backbone()
        return backbone.forward_features(inputs)

    def _extract_tosca_features(self, inputs, task_idx=None):
        if task_idx is not None:
            self._load_tosca(task_idx)

        backbone = self._get_backbone()
        return backbone(inputs)

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
            self._load_tosca(self._cur_task)
            for _, data, label in self.train_loader_for_protonet:
                data = data.to(self._device)
                features = self._extract_tosca_features(data)
                features = self._prepare_gate_features(features)
                collector.update(features, label)

        class_stats = collector.compute_mean_variance()
        self._task_feature_stats[self._cur_task] = class_stats

        logging.info(
            "Collected gate feature statistics for Task %d (%d classes).",
            self._cur_task,
            len(class_stats),
        )

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

            acc = 100.0 * correct / max(total, 1)
            logging.info(
                "Gate Task %d, Epoch %d/%d => Loss %.4f, TaskSelAcc %.2f",
                self._cur_task,
                epoch + 1,
                gate_epochs,
                epoch_loss / max(len(gate_loader), 1),
                acc,
            )

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
        candidate_scores = []

        for task_idx in range(self._cur_task + 1):
            features = self._extract_tosca_features(inputs, task_idx=task_idx)
            features = self._prepare_gate_features(features)
            logits = self._gate(features)
            probs = F.softmax(logits, dim=1)
            candidate_scores.append(probs[:, task_idx].mean().item())

        return int(np.argmax(candidate_scores))

    def _get_selected_task_for_batch(self, inputs, selector=None):
        selector = (selector or self._routing_mode).lower()

        if selector == "gate":
            return self._get_selected_task_for_batch_gate(inputs)

        return self._get_selected_task_for_batch_entropy(inputs)

    def get_routed_logits(self, inputs, selector=None):
        task_idx = self._get_selected_task_for_batch(inputs, selector=selector)
        self._load_tosca(task_idx)
        return self._network(inputs)["logits"]

    def _report_tosca_module_selection(
        self, max_batches_per_task=20, verbose=False, selector="entropy"
    ):
        """
        Report expected vs selected task head for each batch in each true task.
        Also reports classification accuracy with selected head and true (oracle) head.
        """
        self._network.eval()
        selector = selector.lower()
        num_tasks = self._cur_task + 1
        report = {}

        with torch.no_grad():
            for true_task in range(num_tasks):
                loader = self._task_test_loaders[true_task]
                selected_modules = []
                selected_cls_correct = 0
                oracle_cls_correct = 0
                total_samples = 0

                for batch_idx, (_, inputs, targets) in enumerate(loader):
                    if batch_idx >= max_batches_per_task:
                        break

                    inputs = inputs.to(self._device)
                    targets = targets.long().to(self._device)

                    selected_task = self._get_selected_task_for_batch(
                        inputs, selector=selector
                    )
                    selected_modules.append(selected_task)

                    self._load_tosca(selected_task)
                    selected_logits = self._network(inputs)["logits"]
                    selected_preds = torch.argmax(selected_logits, dim=1)
                    selected_cls_correct += (selected_preds == targets).sum().item()

                    self._load_tosca(true_task)
                    oracle_logits = self._network(inputs)["logits"]
                    oracle_preds = torch.argmax(oracle_logits, dim=1)
                    oracle_cls_correct += (oracle_preds == targets).sum().item()

                    total_samples += targets.size(0)

                num_batches = len(selected_modules)
                wrong_batches = [
                    f"B{idx + 1:02d}->T{sel}"
                    for idx, sel in enumerate(selected_modules)
                    if sel != true_task
                ]
                num_correct = num_batches - len(wrong_batches)

                task_match_acc = (
                    100.0 * num_correct / num_batches if num_batches > 0 else 0.0
                )
                selected_cls_acc = (
                    100.0 * selected_cls_correct / total_samples
                    if total_samples > 0
                    else 0.0
                )
                oracle_cls_acc = (
                    100.0 * oracle_cls_correct / total_samples
                    if total_samples > 0
                    else 0.0
                )

                report[true_task] = {
                    "expected_task": true_task,
                    "num_batches": num_batches,
                    "selected_modules": selected_modules,
                    "wrong_batches": wrong_batches,
                    "num_correct": num_correct,
                    "task_match_acc": task_match_acc,
                    "selected_cls_acc": selected_cls_acc,
                    "oracle_cls_acc": oracle_cls_acc,
                }

        if verbose:
            logging.info("=" * 70)
            logging.info(
                f"TOSCA HEAD CHECK REPORT ({selector.upper()}) after Task {self._cur_task}"
            )
            logging.info(
                f"Showing first {max_batches_per_task} test batches for each true task."
            )
            for true_task in range(num_tasks):
                start_cls, end_cls = self._task_ranges[true_task]
                entry = report[true_task]
                selected_line = " ".join(
                    [
                        f"B{idx + 1:02d}:T{mod}"
                        for idx, mod in enumerate(entry["selected_modules"])
                    ]
                )
                logging.info(
                    f"[True Task {true_task}] classes {start_cls}-{end_cls - 1} | expected head: T{true_task}"
                )
                logging.info(
                    f"  Selected heads      : {selected_line if selected_line else 'no batches'}"
                )
                logging.info(
                    f"  Wrong batches       : {', '.join(entry['wrong_batches']) if entry['wrong_batches'] else 'none'}"
                )
                logging.info(
                    f"  Task match          : {entry['num_correct']}/{entry['num_batches']} ({entry['task_match_acc']:.1f}%)"
                )
                logging.info(
                    f"  Class acc (selected): {entry['selected_cls_acc']:.2f}%"
                )
                logging.info(
                    f"  Class acc (oracle)  : {entry['oracle_cls_acc']:.2f}%"
                )
            logging.info("=" * 70)

        return report

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

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []

        selector = self._routing_mode if self._use_gate else "entropy"
        for _, (_, inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(self._device), targets.long().to(self._device)
            with torch.no_grad():
                outputs = self.get_routed_logits(inputs, selector=selector)
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

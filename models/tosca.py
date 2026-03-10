import logging
import numpy as np
import torch
from tqdm import tqdm
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from utils.inc_net import SimpleVitNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy

num_workers = 10


class TaskGate(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=0, num_layers=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.feature_extractor = self._build_feature_extractor()
        last_dim = input_dim if isinstance(self.feature_extractor, nn.Identity) else hidden_dim
        self.classifier = nn.Linear(last_dim, output_dim)

    def _build_feature_extractor(self):
        if self.num_layers <= 1 or self.hidden_dim <= 0:
            return nn.Identity()

        layers = [nn.Linear(self.input_dim, self.hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(self.num_layers - 2):
            layers.extend(
                [nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(inplace=True)]
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.classifier(x)

    def extend_output(self, output_dim):
        if output_dim <= self.classifier.out_features:
            return

        old_classifier = self.classifier
        new_classifier = nn.Linear(old_classifier.in_features, output_dim)
        with torch.no_grad():
            new_classifier.weight[: old_classifier.out_features].copy_(old_classifier.weight)
            new_classifier.bias[: old_classifier.out_features].copy_(old_classifier.bias)
        self.classifier = new_classifier.to(old_classifier.weight.device)


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = SimpleVitNet(args, True)
        self.args = args
        # Store task boundaries: task_ranges[i] = (start_class, end_class)
        self._task_ranges = []
        # Store task-specific test loaders for module-selection reporting
        self._task_test_loaders = []
        # Store module selection reports after each task
        self._tosca_selection_history = []
        self._selection_mode = args.get("selection_mode", "entropy")
        self._enable_gate = args.get("enable_gate", False)
        self._gate = None
        self._class_stats = {}
        self._gate_synthetic_features = []
        self._gate_synthetic_labels = []

        if self._enable_gate:
            self._gate = TaskGate(
                input_dim=self._network.feature_dim,
                output_dim=1,
                hidden_dim=args.get("gate_hidden_dim", 0),
                num_layers=args.get("gate_num_layers", 1),
            ).to(self._device)

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        if self._gate is not None:
            self._gate.extend_output(self._cur_task + 1)

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
            for i, (_, inputs, targets) in enumerate(self.train_loader):
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
        if self._gate is not None:
            self._collect_task_statistics()
            self._generate_gate_synthetic_samples()
            self._train_gate()
            self._save_gate()

        # ============================================================
        # TOSCA MODULE SELECTION REPORT after task training is complete
        # (all TOSCA weights are now saved to disk)
        # ============================================================
        entropy_report = self._report_tosca_module_selection(
            max_batches_per_task=self.args.get("selection_report_batches", 20),
            verbose=True,
            strategy="entropy",
        )
        history_entry = {"task": self._cur_task, "entropy": entropy_report}
        if self._gate is not None:
            history_entry["gate"] = self._report_tosca_module_selection(
                max_batches_per_task=self.args.get("selection_report_batches", 20),
                verbose=True,
                strategy="gate",
            )
        self._tosca_selection_history.append(history_entry)

        self.replace_fc()

    def _collect_task_statistics(self):
        self._network.eval()
        feature_buckets = {}

        with torch.no_grad():
            for _, data, labels in self.train_loader_for_protonet:
                data = data.to(self._device)
                features = self._extract_frozen_features(data).cpu()
                for feature, label in zip(features, labels):
                    class_idx = int(label.item())
                    feature_buckets.setdefault(class_idx, []).append(feature)

        for class_idx, class_features in feature_buckets.items():
            stacked = torch.stack(class_features, dim=0)
            self._class_stats[class_idx] = {
                "task": self._cur_task,
                "mean": stacked.mean(dim=0),
                "std": stacked.std(dim=0, unbiased=False),
                "count": stacked.size(0),
            }

        logging.info(
            "Collected frozen-feature statistics for %d classes after task %d.",
            len(feature_buckets),
            self._cur_task,
        )

    def _generate_gate_synthetic_samples(self):
        samples_per_class = self.args.get("gate_samples_per_class", 1000)
        min_std = self.args.get("gate_min_std", 1e-4)
        task_features = []
        task_labels = []

        for class_idx in range(self._known_classes, self._total_classes):
            stats = self._class_stats[class_idx]
            mean = stats["mean"]
            std = stats["std"].clamp_min(min_std)
            generated = torch.normal(
                mean.expand(samples_per_class, -1),
                std.expand(samples_per_class, -1),
            )
            task_features.append(generated)
            task_labels.append(
                torch.full((samples_per_class,), stats["task"], dtype=torch.long)
            )

        if task_features:
            self._gate_synthetic_features.append(torch.cat(task_features, dim=0))
            self._gate_synthetic_labels.append(torch.cat(task_labels, dim=0))

        logging.info(
            "Generated %d synthetic gate samples for task %d.",
            sum(t.size(0) for t in task_features) if task_features else 0,
            self._cur_task,
        )

    def _train_gate(self):
        if self._gate is None or not self._gate_synthetic_features:
            return

        features = torch.cat(self._gate_synthetic_features, dim=0)
        labels = torch.cat(self._gate_synthetic_labels, dim=0)
        dataset = TensorDataset(features, labels)
        loader = DataLoader(
            dataset,
            batch_size=self.args.get("gate_batch_size", 256),
            shuffle=True,
            num_workers=num_workers,
        )
        optimizer = optim.Adam(
            self._gate.parameters(),
            lr=self.args.get("gate_lr", 1e-3),
            weight_decay=self.args.get("gate_weight_decay", 0.0),
        )

        self._gate.train()
        epochs = self.args.get("gate_train_epochs", 20)
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            for batch_features, batch_labels in loader:
                batch_features = batch_features.to(self._device)
                batch_labels = batch_labels.to(self._device)

                optimizer.zero_grad()
                logits = self._gate(batch_features)
                loss = F.cross_entropy(logits, batch_labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == batch_labels).sum().item()
                total += batch_labels.size(0)

            logging.info(
                "Gate train | task %d | epoch %d/%d | loss %.4f | acc %.2f",
                self._cur_task,
                epoch + 1,
                epochs,
                epoch_loss / max(len(loader), 1),
                100.0 * correct / max(total, 1),
            )

        self._gate.eval()

    def _report_tosca_module_selection(
        self, max_batches_per_task=20, verbose=False, strategy="entropy"
    ):
        """
        Report expected vs selected head for each batch in each true task.
        """
        self._network.eval()
        num_tasks = self._cur_task + 1
        report = {}

        with torch.no_grad():
            for true_task in range(num_tasks):
                loader = self._task_test_loaders[true_task]
                selected_modules = []

                for batch_idx, (_, inputs, _) in enumerate(loader):
                    if batch_idx >= max_batches_per_task:
                        break
                    inputs = inputs.to(self._device)
                    selected_task = self._get_selected_task_for_batch(
                        inputs, strategy=strategy
                    )
                    selected_modules.append(selected_task)

                num_batches = len(selected_modules)
                wrong_batches = [
                    f"B{idx + 1:02d}->T{sel}"
                    for idx, sel in enumerate(selected_modules)
                    if sel != true_task
                ]
                num_correct = num_batches - len(wrong_batches)
                report[true_task] = {
                    "expected_task": true_task,
                    "num_batches": num_batches,
                    "selected_modules": selected_modules,
                    "wrong_batches": wrong_batches,
                    "num_correct": num_correct,
                }

        if verbose:
            logging.info("=" * 70)
            logging.info(
                f"TOSCA HEAD CHECK REPORT after Task {self._cur_task} [{strategy}]"
            )
            logging.info(
                f"Showing first {max_batches_per_task} test batches for each true task."
            )
            for true_task in range(num_tasks):
                start_cls, end_cls = self._task_ranges[true_task]
                entry = report[true_task]
                num_batches = entry["num_batches"]
                selected_line = " ".join(
                    [
                        f"B{idx + 1:02d}:T{mod}"
                        for idx, mod in enumerate(entry["selected_modules"])
                    ]
                )
                if num_batches == 0:
                    acc_str = "N/A"
                else:
                    acc = 100.0 * entry["num_correct"] / num_batches
                    acc_str = f"{entry['num_correct']}/{num_batches} ({acc:.1f}%)"
                logging.info(
                    f"[True Task {true_task}] classes {start_cls}-{end_cls - 1} | expected head: T{true_task}"
                )
                logging.info(
                    f"  Selected heads: {selected_line if selected_line else 'no batches'}"
                )
                logging.info(
                    f"  Wrong batches : {', '.join(entry['wrong_batches']) if entry['wrong_batches'] else 'none'}"
                )
                logging.info(f"  Match         : {acc_str}")
            logging.info("=" * 70)

        return report

    def _extract_frozen_features(self, inputs):
        return self._network.backbone.forward_features(inputs)

    def _get_selected_task_for_batch(self, inputs, strategy=None):
        strategy = strategy or self._selection_mode
        if strategy == "gate" and self._gate is not None:
            with torch.no_grad():
                features = self._extract_frozen_features(inputs)
                gate_logits = self._gate(features)
            return int(torch.argmax(gate_logits.mean(dim=0)).item())

        return self._get_entropy_selected_task_for_batch(inputs)

    def _get_entropy_selected_task_for_batch(self, inputs):
        entropies = []
        for i in range(self._cur_task + 1):
            self._load_tosca(i)
            outputs = self._network(inputs)["logits"]
            probabilities = F.softmax(outputs, dim=1)
            batch_entropy = -torch.sum(
                probabilities * torch.log(probabilities + 1e-9), dim=1
            )
            entropies.append(torch.mean(batch_entropy).item())

        return int(np.argmin(entropies))

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
        return scheduler

    def get_lowentropy_logits(self, inputs):
        return self.get_selected_logits(inputs)

    def get_selected_logits(self, inputs):
        strategy = self._selection_mode
        if strategy == "gate" and self._gate is not None:
            task_idx = self._get_selected_task_for_batch(inputs, strategy="gate")
            self._load_tosca(task_idx)
            return self._network(inputs)["logits"]

        entropies = []
        logits_list = []
        for i in range(self._cur_task + 1):
            self._load_tosca(i)
            outputs = self._network(inputs)["logits"]
            logits_list.append(outputs)
            probabilities = F.softmax(outputs, dim=1)
            batch_entropy = -torch.sum(
                probabilities * torch.log(probabilities + 1e-9), dim=1
            )
            average_entropy = torch.mean(batch_entropy)
            entropies.append(average_entropy.item())
        task_idx = int(np.argmin(entropies))

        return logits_list[task_idx]

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for i, (_, inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(self._device), targets.long().to(self._device)
            with torch.no_grad():
                outputs = self.get_selected_logits(inputs)
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _save_tosca(self):
        path = f"tosca/task{self._cur_task}.pth"
        tosca_state_dict = {
            name: param
            for name, param in self._network.state_dict().items()
            if "tosca" in name
        }
        torch.save(tosca_state_dict, path)
        logging.info(f"tosca parameters saved to {path}.")

    def _save_gate(self):
        if self._gate is None:
            return

        path = f"tosca/gate_task{self._cur_task}.pth"
        torch.save(self._gate.state_dict(), path)
        logging.info(f"gate parameters saved to {path}.")

    def _load_tosca(self, idx):
        path = f"tosca/task{idx}.pth"
        tosca_state_dict = torch.load(path)
        current_state_dict = self._network.state_dict()
        current_state_dict.update(tosca_state_dict)
        self._network.load_state_dict(current_state_dict)

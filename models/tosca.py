import logging
import numpy as np
import torch
import time
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from utils.gating import (
    TaskGate,
    PositionalMoEHead,
    FeatureStatsCollector,
    generate_samples,
)
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

        # Gate components
        self._gate = None
        # task_id -> {class_id: {"mean", "covariance", "count"}} on L2-normalized
        # features (used to train/route the gate).
        self._task_feature_stats = {}
        # Same, but on raw (pre-L2-normalize) features. The toscas are trained on
        # raw features, so head replay must be drawn from these.
        self._task_raw_feature_stats = {}
        self._latest_task_timing = {}
        self._moe_top_k = int(self.args.get("moe_top_k", 1))

        # Joint classifier over concatenated top-k expert features.
        self._moe_head = None
        # idx -> saved tosca state dict (files are immutable once written).
        self._tosca_state_cache = {}

    @property
    def _gate_feature_dim(self):
        return int(self._network.feature_dim)

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
                inputs, targets = (
                    inputs.to(self._device),
                    targets.long().to(self._device),
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

        feature_collection_start = time.perf_counter()
        self._collect_current_task_gate_stats()
        feature_collection_seconds = time.perf_counter() - feature_collection_start
        gate_train_start = time.perf_counter()
        self._train_gate()
        gate_train_seconds = time.perf_counter() - gate_train_start

        moe_head_train_start = time.perf_counter()
        self._train_moe_head()
        moe_head_train_seconds = time.perf_counter() - moe_head_train_start

        replace_fc_start = time.perf_counter()
        self.replace_fc()
        replace_fc_seconds = time.perf_counter() - replace_fc_start
        task_train_seconds = time.perf_counter() - task_train_start

        self._latest_task_timing = {
            "backbone_train_seconds": backbone_train_seconds,
            "feature_collection_seconds": feature_collection_seconds,
            "gate_train_seconds": gate_train_seconds,
            "moe_head_train_seconds": moe_head_train_seconds,
            "replace_fc_seconds": replace_fc_seconds,
            "task_train_seconds": task_train_seconds,
        }
        logging.info(
            "Task {} timing => backbone_train {:.2f}s, feature_collection {:.2f}s, gate_train {:.2f}s, moe_head_train {:.2f}s, replace_fc {:.2f}s, total_train {:.2f}s".format(
                self._cur_task,
                backbone_train_seconds,
                feature_collection_seconds,
                gate_train_seconds,
                moe_head_train_seconds,
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
        # Tosca + fc trainable every task.
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

    def _synthetic_per_class(self):
        return int(self.args.get("synthetic_per_class", 200))

    def _gate_min_variance(self):
        return float(self.args.get("gate_min_variance", 1e-6))

    def _moe_stats_mode(self):
        return str(self.args.get("moe_stats_mode", "covariance"))

    def _moe_synthetic_per_class(self):
        return int(self.args.get("moe_synthetic_per_class", self._synthetic_per_class()))

    def _classes_per_task(self):
        start, end = self._task_ranges[0]
        return end - start

    def _collect_current_task_gate_stats(self):
        """Store per-class mean + full covariance of real ViT features for the
        current task. The gate is later trained on synthetic samples drawn from
        these Gaussians; tosca heads keep training on real features."""
        self._network.eval()
        collector = FeatureStatsCollector(
            feature_dim=self._gate_feature_dim,
            min_variance=self._gate_min_variance(),
            stats_mode="covariance",
        )
        raw_collector = FeatureStatsCollector(
            feature_dim=self._gate_feature_dim,
            min_variance=self._gate_min_variance(),
            stats_mode=self._moe_stats_mode(),
        )
        with torch.no_grad():
            for _, data, label in self.train_loader_for_protonet:
                data = data.to(self._device)
                label = label.long()
                features = self._extract_backbone_features(data)
                raw_collector.update(features, label)
                norm_features = self._prepare_gate_features(features)
                collector.update(norm_features, label)

        class_stats = collector.compute_mean_variance()
        self._task_feature_stats[self._cur_task] = class_stats
        self._task_raw_feature_stats[self._cur_task] = (
            raw_collector.compute_mean_variance()
        )
        logging.info(
            "Stored full-covariance gate stats for task %s (%s classes).",
            self._cur_task,
            len(class_stats),
        )

    def _collect_all_gate_features(self):
        """Generate synthetic ViT features from every task's stored per-class
        Gaussians, labeled by task id, for gate training."""
        n_samples = self._synthetic_per_class()
        min_variance = self._gate_min_variance()

        all_features = []
        all_targets = []
        for task_idx in range(self._cur_task + 1):
            class_stats = self._task_feature_stats.get(task_idx)
            if not class_stats:
                continue
            sampled_features, _ = generate_samples(
                class_stats,
                n_samples=n_samples,
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
                torch.empty(0, self._gate_feature_dim),
                torch.empty(0, dtype=torch.long),
            )

        return torch.cat(all_features, dim=0), torch.cat(all_targets, dim=0)

    def _init_or_extend_gate(self):
        num_tasks = self._cur_task + 1
        if self._gate is None:
            self._gate = TaskGate(
                input_dim=self._gate_feature_dim,
                num_tasks=num_tasks,
                hidden_dim=int(self.args.get("gate_hidden_dim", 0)),
                dropout=float(self.args.get("gate_dropout", 0.0)),
            ).to(self._device)
        else:
            self._gate.extend(num_tasks)
            self._gate.to(self._device)

    def _train_gate(self):
        self._init_or_extend_gate()

        train_x, train_y = self._collect_all_gate_features()

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

        assert self._gate is not None
        if self._gate.num_tasks < 2:
            logging.info(
                "Gate Task 0: single task, skipping training (nothing to discriminate)."
            )
            self._gate.eval()
            self._save_gate()
            return

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
                "input_dim": self._gate_feature_dim,
                "hidden_dim": int(self.args.get("gate_hidden_dim", 0)),
                "dropout": float(self.args.get("gate_dropout", 0.0)),
            },
            path,
        )
        logging.info(f"Gate parameters saved to {path}.")

    def _generate_head_replay_raw(self):
        """Synthetic raw (pre-normalize) ViT features for every class seen so
        far, labeled by GLOBAL class id, drawn from the stored per-class
        Gaussians. These are pushed through the toscas to train the head."""
        n_samples = self._moe_synthetic_per_class()
        min_variance = self._gate_min_variance()

        all_features = []
        all_labels = []
        for task_idx in range(self._cur_task + 1):
            class_stats = self._task_raw_feature_stats.get(task_idx)
            if not class_stats:
                continue
            feats, labels = generate_samples(
                class_stats,
                n_samples=n_samples,
                min_variance=min_variance,
                device="cpu",
            )
            if feats.numel() == 0:
                continue
            all_features.append(feats)
            all_labels.append(labels)

        if len(all_features) == 0:
            return (
                torch.empty(0, self._gate_feature_dim),
                torch.empty(0, dtype=torch.long),
            )
        return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)

    def _build_moe_training_tensors(self):
        """Turn replayed raw features into (concatenated expert features,
        positional target) pairs for head training. The gate (frozen) picks the
        top-k experts per sample; each selected tosca produces that sample's
        slot feature; the target is the slot+local-class index of the sample's
        true task (ignored when the true task is not in the top-k)."""
        raw_x, class_y = self._generate_head_replay_raw()
        if raw_x.numel() == 0:
            return None

        num_tasks = self._cur_task + 1
        top_k = self._moe_top_k
        eff_k = max(1, min(top_k, num_tasks))
        inc = self._classes_per_task()
        feature_dim = self._gate_feature_dim
        n = raw_x.size(0)
        chunk = 4096

        starts = torch.tensor([s for s, _ in self._task_ranges])
        true_task = torch.zeros(n, dtype=torch.long)
        for t, (start, end) in enumerate(self._task_ranges):
            true_task[(class_y >= start) & (class_y < end)] = t

        assert self._gate is not None
        self._gate.eval()
        raw_x_dev = raw_x.to(self._device)

        # Frozen-gate top-k routing for every replay sample.
        topk_chunks = []
        with torch.no_grad():
            for i in range(0, n, chunk):
                xb = raw_x_dev[i : i + chunk]
                gate_logits = self._gate(self._prepare_gate_features(xb))
                topk_chunks.append(torch.topk(gate_logits, eff_k, dim=1).indices.cpu())
        topk_tasks = torch.cat(topk_chunks, dim=0)  # [N, eff_k]

        # Per-sample slot features. Load each tosca once, fill the slots that
        # selected it. Slots >= eff_k stay zero-padded.
        slot_feats = torch.zeros(n, top_k, feature_dim)
        with torch.no_grad():
            for t in range(num_tasks):
                sel = topk_tasks == t  # [N, eff_k]
                rows = sel.any(dim=1)
                if not rows.any():
                    continue
                self._load_tosca(t)
                idx = rows.nonzero(as_tuple=True)[0]
                ft_full = torch.zeros(n, feature_dim)
                for i in range(0, idx.numel(), chunk):
                    rows_chunk = idx[i : i + chunk]
                    ft_full[rows_chunk] = (
                        self._get_backbone()
                        .forward_tosca(raw_x_dev[rows_chunk])
                        .cpu()
                    )
                for s in range(eff_k):
                    slot_mask = sel[:, s]
                    if slot_mask.any():
                        slot_feats[slot_mask, s, :] = ft_full[slot_mask]

        match = topk_tasks == true_task.unsqueeze(1)  # [N, eff_k]
        has_true = match.any(dim=1)
        slot_idx = torch.argmax(match.int(), dim=1)  # first matching slot
        local_class = class_y - starts[true_task]
        target = slot_idx * inc + local_class
        target[~has_true] = -100  # unrecoverable: true task not routed

        return slot_feats.reshape(n, top_k * feature_dim), target, has_true

    def _train_moe_head(self):
        num_tasks = self._cur_task + 1
        top_k = self._moe_top_k
        inc = self._classes_per_task()
        feature_dim = self._gate_feature_dim

        if self._moe_head is None:
            self._moe_head = PositionalMoEHead(
                feature_dim=feature_dim,
                top_k=top_k,
                classes_per_task=inc,
                hidden_dim=int(self.args.get("moe_hidden_dim", 1024)),
                dropout=float(self.args.get("moe_dropout", 0.1)),
            ).to(self._device)

        # Fresh synthetic draw every epoch: the head then learns the population
        # decision boundary of the per-class Gaussians instead of memorizing one
        # finite draw, which is the main lever for closing the synthetic->real
        # gap while staying fully synthetic. Disable to reuse a single draw.
        resample = bool(self.args.get("moe_resample_each_epoch", True))
        batch_size = int(self.args.get("moe_batch_size", 512))

        def build_loader():
            tensors = self._build_moe_training_tensors()
            if tensors is None:
                return None, None
            train_x, train_y, has_true = tensors
            loader = DataLoader(
                TensorDataset(train_x, train_y),
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
            )
            return loader, has_true

        loader, has_true = build_loader()
        if loader is None:
            logging.warning("No replay features available for MoE head training.")
            return
        logging.info(
            "MoE head Task %s: %d replay samples/epoch, true task routed in top-%d "
            "for %.2f%% (resample_each_epoch=%s).",
            self._cur_task,
            len(loader.dataset),
            max(1, min(top_k, num_tasks)),
            100.0 * has_true.float().mean().item(),
            resample,
        )

        optimizer = optim.Adam(
            self._moe_head.parameters(),
            lr=float(self.args.get("moe_lr", 1e-3)),
            weight_decay=float(self.args.get("moe_weight_decay", 1e-4)),
        )
        criterion = torch.nn.CrossEntropyLoss(
            ignore_index=-100,
            label_smoothing=float(self.args.get("moe_label_smoothing", 0.0)),
        )
        epochs = int(self.args.get("moe_epochs", 30))

        self._moe_head.train()
        for epoch in range(epochs):
            if resample and epoch > 0:
                loader, _ = build_loader()
            epoch_loss = 0.0
            correct = 0
            total = 0
            for features, targets in loader:
                features = features.to(self._device)
                targets = targets.to(self._device)

                optimizer.zero_grad()
                logits = self._moe_head(features)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                valid = targets != -100
                if valid.any():
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds[valid] == targets[valid]).sum().item()
                    total += int(valid.sum().item())

            avg_loss = epoch_loss / max(len(loader), 1)
            train_acc = 100.0 * correct / total if total > 0 else 0.0
            logging.info(
                f"MoE Head Task {self._cur_task}, Epoch {epoch + 1}/{epochs} => "
                f"Loss {avg_loss:.4f}, Train_accy {train_acc:.2f}"
            )

        self._moe_head.eval()
        self._save_moe_head()

    def _save_moe_head(self):
        if self._moe_head is None:
            return
        path = f"tosca/moe_head_task{self._cur_task}.pth"
        torch.save(
            {
                "state_dict": self._moe_head.state_dict(),
                "feature_dim": self._gate_feature_dim,
                "top_k": self._moe_top_k,
                "classes_per_task": self._classes_per_task(),
                "hidden_dim": int(self.args.get("moe_hidden_dim", 1024)),
                "dropout": float(self.args.get("moe_dropout", 0.1)),
            },
            path,
        )
        logging.info(f"MoE head parameters saved to {path}.")

    def _get_moe_routed_logits(self, inputs, oracle_tasks=None):
        """Top-k mixture-of-experts routing with a JOINT head. The gate selects
        the top-k tasks per sample; each selected task's tosca produces that
        sample's slot feature; the slot features are concatenated (gate-rank
        order, zero-padded) and a single PositionalMoEHead scores them jointly.
        The head's per-slot class scores are scattered back to the selected
        tasks' (disjoint) global class blocks, and argmax over that union is the
        prediction. k = moe_top_k.

        If ``oracle_tasks`` ([B] long tensor of true task ids) is given, each
        sample's true task is guaranteed to be in the routed set (swapped into
        the lowest-ranked slot when the gate missed it). This forces routing
        recall@k = 100%, isolating the head's within-set accuracy from the
        gate's recall loss."""
        if self._gate is None or self._gate.num_tasks < (self._cur_task + 1):
            raise RuntimeError(
                "Gate is not initialized or out of sync with current task count."
            )
        if self._moe_head is None:
            raise RuntimeError("MoE head is not initialized.")

        self._gate.eval()
        self._moe_head.eval()
        vit_features = self._extract_backbone_features(inputs)
        gate_features = self._prepare_gate_features(vit_features)
        gate_logits = self._gate(gate_features)

        num_tasks = self._cur_task + 1
        top_k = self._moe_top_k
        eff_k = max(1, min(top_k, num_tasks))
        inc = self._classes_per_task()
        feature_dim = self._gate_feature_dim
        batch_size = inputs.size(0)
        topk_tasks = torch.topk(gate_logits, eff_k, dim=1).indices  # [B, eff_k]

        # Oracle routing: guarantee the true task is in the routed set so that
        # recall@k = 100%. Only override samples the gate already missed, and
        # place the true task in the lowest-ranked slot to perturb the gate's
        # top choices as little as possible.
        if oracle_tasks is not None:
            oracle_tasks = oracle_tasks.to(topk_tasks.device)
            missing = ~(topk_tasks == oracle_tasks.unsqueeze(1)).any(dim=1)
            if missing.any():
                topk_tasks[missing, eff_k - 1] = oracle_tasks[missing]

        # Build the [B, top_k, D] concatenated expert features (zero-padded).
        slot_feats = torch.zeros(
            batch_size, top_k, feature_dim, device=inputs.device
        )
        for task_idx in torch.unique(topk_tasks).tolist():
            task_idx = int(task_idx)
            sel = topk_tasks == task_idx  # [B, eff_k]
            self._load_tosca(task_idx)
            task_features = self._get_backbone().forward_tosca(vit_features)
            for s in range(eff_k):
                slot_mask = sel[:, s]
                if slot_mask.any():
                    slot_feats[slot_mask, s, :] = task_features[slot_mask]

        head_logits = self._moe_head(
            slot_feats.reshape(batch_size, top_k * feature_dim)
        ).view(batch_size, top_k, inc)

        # Scatter each slot's class scores into its task's global class block.
        total_classes = self._network.fc.out_features
        out_logits = torch.full(
            (batch_size, total_classes), float("-inf"), device=inputs.device
        )
        starts = torch.tensor(
            [s for s, _ in self._task_ranges], device=inputs.device
        )
        offsets = torch.arange(inc, device=inputs.device)
        for s in range(eff_k):
            block_start = starts[topk_tasks[:, s]]  # [B]
            idx = block_start.unsqueeze(1) + offsets.unsqueeze(0)  # [B, inc]
            out_logits.scatter_(1, idx, head_logits[:, s, :])

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
        recall_k_correct = 0
        total = 0
        k = max(1, min(self._moe_top_k, len(self._task_ranges)))
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
                gate_logits = self._gate(features)
                chosen_task = torch.argmax(gate_logits, dim=1).cpu()
                topk_tasks = torch.topk(gate_logits, k, dim=1).indices.cpu()  # [B, k]

                true_task = torch.zeros_like(targets)
                for t, (start, end) in enumerate(self._task_ranges):
                    mask = (targets >= start) & (targets < end)
                    true_task[mask] = t

                correct += (chosen_task == true_task).sum().item()
                recall_k_correct += (
                    (topk_tasks == true_task.unsqueeze(1)).any(dim=1).sum().item()
                )
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
        recall_at_k = 100.0 * recall_k_correct / total if total > 0 else 0.0
        for task_idx, stats in per_task.items():
            task_total = stats["total"]
            stats["accuracy"] = (
                100.0 * stats["correct"] / task_total if task_total > 0 else 0.0
            )
        return {
            "top1": gate_acc,
            "recall_at_k": recall_at_k,
            "top_k": k,
            "per_task": per_task,
        }

    def _compute_gate_routing_flops(self, batch_size=1):
        input_dim = int(self._gate_feature_dim)
        hidden_dim = int(self.args.get("gate_hidden_dim", 0))
        num_tasks = max(self._cur_task + 1, 1)

        if hidden_dim > 0:
            hidden_flops = 2 * input_dim * hidden_dim + hidden_dim
            classifier_flops = 2 * hidden_dim * num_tasks
        else:
            hidden_flops = 0
            classifier_flops = 2 * input_dim * num_tasks

        per_sample_flops = hidden_flops + classifier_flops
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
        oracle_pred, oracle_true = self._eval_cnn(self.test_loader, oracle=True)
        oracle_accy = self._evaluate(oracle_pred, oracle_true)
        gate_accy = self._eval_gate_routing(self.test_loader)
        if gate_accy is not None:
            gate_accy["oracle_top1"] = oracle_accy["top1"]
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

    def _true_task_from_targets(self, targets):
        """Map global class labels [B] to their task ids [B] via task ranges."""
        true_task = torch.zeros_like(targets)
        for t, (start, end) in enumerate(self._task_ranges):
            true_task[(targets >= start) & (targets < end)] = t
        return true_task

    def _eval_cnn(self, loader, oracle=False):
        self._network.eval()
        y_pred, y_true = [], []

        for _, (_, inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(self._device), targets.long().to(self._device)
            with torch.no_grad():
                oracle_tasks = (
                    self._true_task_from_targets(targets) if oracle else None
                )
                outputs = self._get_moe_routed_logits(
                    inputs, oracle_tasks=oracle_tasks
                )
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[1]
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
        # tosca/task{idx}.pth is written once and never modified, so the loaded
        # state can be cached to avoid repeated disk reads (hot in per-epoch
        # head resampling and per-batch eval routing).
        tosca_state_dict = self._tosca_state_cache.get(idx)
        if tosca_state_dict is None:
            tosca_state_dict = torch.load(
                f"tosca/task{idx}.pth", map_location=self._device
            )
            self._tosca_state_cache[idx] = tosca_state_dict
        current_state_dict = self._network.state_dict()
        current_state_dict.update(tosca_state_dict)
        self._network.load_state_dict(current_state_dict)

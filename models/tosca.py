import logging
import os
import re
import numpy as np
import torch
import time
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

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
        self._latest_task_timing = {}

        # idx -> saved tosca state dict (files are immutable once written).
        self._tosca_state_cache = {}

        # Ridge classifiers (RanPAC-style decorrelation). Shared random
        # projection P + Gram/class-sum accumulators, held IN MEMORY for the
        # whole run so any previously routed task can still be scored.
        # Inference (ridge_scope "global_router"): the global ridge on the
        # frozen features routes top-1 to a task, then that task's
        # TOSCA-expert ridge head classifies within the task.
        self._ridge_scope = str(self.args.get("ridge_scope", "global_router"))
        if not self.args.get("use_ridge", False) or self._ridge_scope != "global_router":
            raise ValueError(
                "Config must set use_ridge=true with ridge_scope 'global_router' "
                f"(got use_ridge={self.args.get('use_ridge')}, "
                f"ridge_scope={self._ridge_scope!r})."
            )
        # G/C are only ever needed in-memory for the current process (the
        # live run keeps every task's G/C around so any previously routed
        # task can still be scored). Disk persistence saves just the SOLVED
        # W (a few KB) for reproducibility/inspection -- not the raw G/C
        # (a dense [M, M] Gram matrix, ~900MB at M=15000 PER TASK; a 20-task
        # dataset x 5 seeds at raw G/C would need ~100GB).
        self._ridge_P = None  # [feat_dim, M] fixed random projection
        self._ridge_G = {}  # task_idx -> [M, M]  (Phi^T Phi)
        self._ridge_C = {}  # task_idx -> [M, inc] (Phi^T Y_onehot)
        self._ridge_W_cache = {}  # (task_idx, lambda) -> [M, inc]
        # Global-scope running accumulators (shared frozen-feature space).
        self._ridge_G_global = None  # [M, M]
        self._ridge_C_global = None  # [M, total_classes]

    def incremental_train(self, data_manager):
        self._setup_task_loaders(data_manager)
        self._train()

    def _setup_task_loaders(self, data_manager):
        """Bookkeeping + dataloaders for the current task, split out from
        incremental_train so an offline script can replay task boundaries
        (task_ranges, fc sizing, test_loader) without running gradient
        training -- it loads saved adapters/ridge state instead."""
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
        if self._cur_task == 0:
            self._save_adaptmlp()

        replace_fc_start = time.perf_counter()
        self.replace_fc()
        self._save_ridge_state()
        replace_fc_seconds = time.perf_counter() - replace_fc_start
        task_train_seconds = time.perf_counter() - task_train_start

        self._latest_task_timing = {
            "backbone_train_seconds": backbone_train_seconds,
            "replace_fc_seconds": replace_fc_seconds,
            "task_train_seconds": task_train_seconds,
        }
        logging.info(
            "Task {} timing => backbone_train {:.2f}s, replace_fc {:.2f}s, total_train {:.2f}s".format(
                self._cur_task,
                backbone_train_seconds,
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

    def _classes_per_task(self):
        start, end = self._task_ranges[0]
        return end - start

    # ------------------------------------------------------------------ ridge
    def _ridge_projection(self, feature_dim):
        """Fixed shared random projection [feat_dim -> M] (RanPAC-style lift).
        Created once, seeded for reproducibility."""
        if self._ridge_P is None:
            M = int(self.args.get("ridge_proj_dim", 5000))
            gen = torch.Generator(device="cpu").manual_seed(1993)
            P = torch.randn(feature_dim, M, generator=gen)
            self._ridge_P = P.to(self._device)
        return self._ridge_P

    def _ridge_features(self, feats):
        """L2-normalize (optional), project, ReLU -> [N, M]."""
        if bool(self.args.get("ridge_normalize", True)):
            feats = F.normalize(feats, p=2, dim=1)
        phi = feats @ self._ridge_projection(feats.size(1))
        if str(self.args.get("ridge_activation", "relu")) == "relu":
            phi = F.relu(phi)
        return phi

    def _fit_ridge_head(self, embeddings, labels):
        """Accumulate this task's Gram G_t = Phi^T Phi and class-sum
        C_t = Phi^T Y (one-hot over the task's own classes). Storing G_t/C_t
        (not W_t) keeps lambda re-solvable at eval with no retraining."""
        inc = self._classes_per_task()
        start = self._task_ranges[self._cur_task][0]
        embeddings = embeddings.to(self._device)
        local = (labels.to(self._device) - start).long()
        phi = self._ridge_features(embeddings)  # [N, M]
        onehot = torch.zeros(phi.size(0), inc, device=self._device)
        onehot[torch.arange(phi.size(0), device=self._device), local] = 1.0
        self._ridge_G[self._cur_task] = (phi.t() @ phi).float()
        self._ridge_C[self._cur_task] = (phi.t() @ onehot).float()
        self._ridge_W_cache.clear()  # invalidate cached solves
        logging.info(
            "Ridge head Task %s fit: M=%d, N=%d, classes=%d.",
            self._cur_task,
            phi.size(1),
            phi.size(0),
            inc,
        )

    def ridge_extra_param_count(self):
        """Element count of the ridge classifier weights that live outside
        self._network (the random projection P and every solved head),
        so count_parameters(self._network) can be corrected to reflect the
        model's true total size. Unlike RanPAC -- whose equivalent solved
        ridge weight replaces self._network.fc.weight and is therefore
        already counted -- TOSCA's ridge weights are plain Learner
        attributes/dict entries, invisible to count_parameters().

        Computed from _ridge_C/_ridge_C_global (persisted for every seen
        task) rather than the lazily-populated _ridge_W_cache, since the
        cache only holds whichever (task, lambda) pairs eval actually
        solved for and may not cover every task after a cache-clearing
        _fit_ridge_head call.
        """
        extra = 0
        if self._ridge_P is not None:
            extra += self._ridge_P.numel()
        for c in self._ridge_C.values():
            extra += c.numel()
        if self._ridge_C_global is not None:
            extra += self._ridge_C_global.numel()
        return extra

    def _ridge_weight(self, task_idx, lam):
        """W_t = (G_t + lam I)^-1 C_t, cached per (task, lambda). Solved in
        float64 for conditioning, returned float32."""
        key = (task_idx, float(lam))
        W = self._ridge_W_cache.get(key)
        if W is None:
            G = self._ridge_G[task_idx].double()
            C = self._ridge_C[task_idx].double()
            eye = torch.eye(G.size(0), device=G.device, dtype=G.dtype)
            W = torch.linalg.solve(G + float(lam) * eye, C).float()
            self._ridge_W_cache[key] = W
        return W

    def _routed_ridge_from_tasks(
        self, vit_features, topk_tasks, lam, oracle_tasks=None
    ):
        """Given shared ViT features and a per-sample set of candidate tasks
        [B, k] (from the global-ridge router), score each routed expert's
        classes with its per-task ridge head and scatter to the global class
        union. Argmax over the union is a JOINT decision -- ridge regresses to
        one-hot, so scores are cross-expert comparable."""
        inc = self._classes_per_task()
        device = vit_features.device
        batch_size = vit_features.size(0)
        total_classes = self._network.fc.out_features
        starts = torch.tensor([s for s, _ in self._task_ranges], device=device)
        offsets = torch.arange(inc, device=device)

        out_logits = torch.full(
            (batch_size, total_classes), float("-inf"), device=device
        )
        for task_idx in torch.unique(topk_tasks).tolist():
            task_idx = int(task_idx)
            if task_idx not in self._ridge_G:
                continue
            row_ids = (topk_tasks == task_idx).any(dim=1).nonzero(as_tuple=True)[0]
            self._load_tosca(task_idx)
            feats = self._get_backbone().forward_tosca(vit_features[row_ids])
            scores = self._ridge_features(feats) @ self._ridge_weight(task_idx, lam)
            start = int(starts[task_idx].item())
            out_logits[row_ids, start : start + inc] = scores

        if oracle_tasks is not None:
            oracle_tasks = oracle_tasks.to(device)
            routed_true = (topk_tasks == oracle_tasks.unsqueeze(1)).any(dim=1)
            block = starts[oracle_tasks].unsqueeze(1) + offsets.unsqueeze(0)
            keep = torch.full((batch_size, total_classes), float("-inf"), device=device)
            keep.scatter_(1, block, out_logits.gather(1, block))
            out_logits = torch.where(routed_true.unsqueeze(1), keep, out_logits)
        return out_logits

    def _task_scores_from_logits(self, logits):
        """Class logits [B, C] -> per-task score [B, num_tasks] (max class in
        each task's block). Used to route from the global ridge's own output."""
        device = logits.device
        scores = torch.full(
            (logits.size(0), len(self._task_ranges)), float("-inf"), device=device
        )
        for t, (start, end) in enumerate(self._task_ranges):
            scores[:, t] = logits[:, start:end].max(dim=1).values
        return scores

    def _get_global_routed_ridge_logits(self, inputs, oracle_tasks=None, lam=None):
        """global_router: the GLOBAL ridge routes, then the per-task TOSCA
        expert ridge classifies within the chosen task. Top-1 routing --
        validated best over k=1/2/5 (larger k only injects impostor experts)."""
        if lam is None:
            lam = float(self.args.get("ridge_lambda", 1e4))
        vit_features = self._extract_backbone_features(inputs)
        global_logits = self._ridge_features(vit_features) @ self._global_ridge_weight(
            lam
        )
        task_scores = self._task_scores_from_logits(global_logits)
        top1_task = task_scores.argmax(dim=1, keepdim=True)  # [B, 1]
        return self._routed_ridge_from_tasks(vit_features, top1_task, lam, oracle_tasks)

    def _accumulate_global_ridge(self, frozen, labels):
        """RanPAC-style single classifier: accumulate one shared Gram
        G = Phi^T Phi and class-sum C = Phi^T Y over ALL classes, in the frozen
        (pre-tosca) feature space. Decorrelates across tasks. Grows C's
        columns as new classes arrive."""
        total = self._total_classes
        frozen = frozen.to(self._device)
        labels = labels.to(self._device).long()
        phi = self._ridge_features(frozen)  # [N, M]
        onehot = torch.zeros(phi.size(0), total, device=self._device)
        onehot[torch.arange(phi.size(0), device=self._device), labels] = 1.0
        G = (phi.t() @ phi).float()
        C = (phi.t() @ onehot).float()
        if self._ridge_G_global is None:
            self._ridge_G_global = G
            self._ridge_C_global = C
        else:
            self._ridge_G_global = self._ridge_G_global + G
            prev = self._ridge_C_global
            grown = torch.zeros(prev.size(0), total, device=self._device)
            grown[:, : prev.size(1)] = prev
            self._ridge_C_global = grown + C
        self._ridge_W_cache.clear()
        logging.info(
            "Global ridge accumulated Task %s: M=%d, N=%d, classes=%d.",
            self._cur_task,
            phi.size(1),
            phi.size(0),
            total,
        )

    def _global_ridge_weight(self, lam):
        key = ("__global__", float(lam))
        W = self._ridge_W_cache.get(key)
        if W is None:
            G = self._ridge_G_global.double()
            C = self._ridge_C_global.double()
            eye = torch.eye(G.size(0), device=G.device, dtype=G.dtype)
            W = torch.linalg.solve(G + float(lam) * eye, C).float()
            self._ridge_W_cache[key] = W
        return W

    def after_task(self):
        self._network.backbone.reset_tosca()
        self._known_classes = self._total_classes

    def replace_fc(self):
        self._network.eval()
        self._load_tosca(self._cur_task)
        embedding_list = []
        frozen_list = []
        label_list = []
        with torch.no_grad():
            for _, data, label in self.train_loader_for_protonet:
                data = data.to(self._device)
                label = label.long().to(self._device)
                vit = self._extract_backbone_features(data)
                frozen_list.append(vit.cpu())
                embedding_list.append(self._get_backbone().forward_tosca(vit).cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        class_list = np.unique(self.train_dataset.labels)
        for class_index in class_list:
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            embedding = embedding_list[data_index]
            proto = embedding.mean(0)
            self._network.fc.weight.data[class_index] = proto

        # Fit ridge from the same real features -- no extra data pass,
        # exemplar-free (only Gram/class-sum stored, never raw samples).
        # global_router needs BOTH: the global head (routes) and the per-task
        # heads (classify within the routed task).
        self._accumulate_global_ridge(torch.cat(frozen_list, dim=0), label_list)
        self._fit_ridge_head(embedding_list, label_list)

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

    def _compute_routing_flops(self, batch_size=1):
        num_tasks = max(self._cur_task + 1, 1)

        # Global ridge as router/classifier: normalize + random projection
        # (768->M) + ReLU + linear head to all seen classes. (In global_router
        # the per-task expert cost is separate.)
        feat_dim = int(self._network.feature_dim)
        M = int(self.args.get("ridge_proj_dim", 5000))
        classes = int(self._total_classes)
        per_sample_flops = feat_dim + 2 * feat_dim * M + M + 2 * M * classes

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
        eval_seconds = time.perf_counter() - eval_start
        num_samples = len(self.test_loader.dataset)
        routing_flops = self._compute_routing_flops(
            batch_size=self.test_loader.batch_size
        )
        metrics = {}
        metrics["eval_seconds"] = eval_seconds
        metrics["ms_per_sample"] = 1000.0 * eval_seconds / max(num_samples, 1)
        metrics["routing_flops"] = routing_flops
        return cnn_accy, nme_accy, metrics

    def _true_task_from_targets(self, targets):
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
                oracle_tasks = self._true_task_from_targets(targets) if oracle else None
                outputs = self._get_global_routed_ridge_logits(
                    inputs, oracle_tasks=oracle_tasks
                )
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)

    def _ckpt_dir(self):
        """Namespace all persisted checkpoints (tosca/adaptmlp/ridge) by
        dataset + prefix + seed so concurrent or sequential runs -- different
        datasets, hyperparameter grid variants, or the 5-seed benchmark runs
        of the SAME dataset -- never read/write each other's files. Without
        the seed tag, seed 1994 would silently overwrite seed 1993's saved
        adapters/ridge matrices, breaking offline sweeps and resumes."""
        dataset = str(self.args.get("dataset", "data"))
        prefix = re.sub(
            r"[^A-Za-z0-9_.-]+", "_", str(self.args.get("prefix", "")).strip()
        )
        tag = f"{dataset}__{prefix}" if prefix else dataset
        seed = self.args.get("seed")
        if isinstance(seed, (int, np.integer)):
            tag = f"{tag}__seed{int(seed)}"
        path = os.path.join("tosca", tag)
        os.makedirs(path, exist_ok=True)
        return path

    def _save_adaptmlp(self):
        """ViT AdaptMLP adapters are only trained on task 0 (_set_trainable),
        then frozen forever -- but _save_tosca's 'tosca' name filter doesn't
        catch them, so without this they'd never be persisted and an offline
        script couldn't rebuild the exact frozen feature space training used."""
        state = {
            name: param
            for name, param in self._network.state_dict().items()
            if "adaptmlp" in name
        }
        path = os.path.join(self._ckpt_dir(), "adaptmlp.pth")
        torch.save(state, path)
        logging.info("AdaptMLP parameters saved to %s.", path)

    def _load_adaptmlp(self):
        path = os.path.join(self._ckpt_dir(), "adaptmlp.pth")
        state = torch.load(path, map_location=self._device)
        current_state_dict = self._network.state_dict()
        current_state_dict.update(state)
        self._network.load_state_dict(current_state_dict)

    def _save_ridge_state(self):
        """Persist the SOLVED ridge weights W (not the raw G/C Gram
        matrices) for reproducibility/inspection -- a few KB total, vs. G
        being a dense [M, M] matrix (~900MB at M=15000 PER TASK). The global
        snapshot uses a single fixed filename since only the latest one is
        ever needed once training has moved past a task."""
        t = self._cur_task
        ckpt_dir = self._ckpt_dir()
        lam = float(self.args.get("ridge_lambda", 1e4))
        if t in self._ridge_G:
            torch.save(
                {"W": self._ridge_weight(t, lam).cpu(), "lambda": lam},
                os.path.join(ckpt_dir, f"ridge_task{t}.pth"),
            )
        if self._ridge_G_global is not None:
            torch.save(
                {"W": self._global_ridge_weight(lam).cpu(), "lambda": lam},
                os.path.join(ckpt_dir, "ridge_global.pth"),
            )
        logging.info("Ridge state (solved W) saved for task %s.", t)

    def _save_tosca(self):
        path = os.path.join(self._ckpt_dir(), f"task{self._cur_task}.pth")
        tosca_state_dict = {
            name: param
            for name, param in self._network.state_dict().items()
            if "tosca" in name
        }
        torch.save(tosca_state_dict, path)
        logging.info(f"tosca parameters saved to {path}.")

    def _load_tosca(self, idx):
        # task{idx}.pth is written once and never modified, so the loaded
        # state can be cached to avoid repeated disk reads (hot in per-epoch
        # head resampling and per-batch eval routing).
        tosca_state_dict = self._tosca_state_cache.get(idx)
        if tosca_state_dict is None:
            tosca_state_dict = torch.load(
                os.path.join(self._ckpt_dir(), f"task{idx}.pth"),
                map_location=self._device,
            )
            self._tosca_state_cache[idx] = tosca_state_dict
        current_state_dict = self._network.state_dict()
        current_state_dict.update(tosca_state_dict)
        self._network.load_state_dict(current_state_dict)

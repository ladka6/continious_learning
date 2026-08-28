"""Prism: global ridge router (random projections) + per-task TOSCA-adapted
CosineLinear expert heads. "TOSCA" below refers specifically to the paper's
adapter block (backbone/vit_tosca.py) that Prism uses internally -- the
router, decoupled per-task heads, and their combination are Prism's own
contribution, not the original TOSCA paper's architecture."""

import logging
import os
import re
import numpy as np
import torch
import time
from tqdm import tqdm
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from utils.inc_net import SimpleVitNet
from backbone.linears import CosineLinear
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
        # idx -> saved expert-head state dict (files are immutable once written).
        self._expert_head_cache = {}
        # Running element count of every saved per-task expert head's
        # parameters, for ridge_extra_param_count() -- these heads are
        # standalone CosineLinear modules, never registered inside
        # self._network, so count_parameters(self._network) can't see them.
        self._expert_param_total = 0
        # Trainable param count of the CURRENT task's expert head at the
        # moment its training loop finished (see _save_expert_head) --
        # exposed via expert_head_trainable_param_count() so trainer.py can
        # correct the "trainable_params" metric it logs right after this
        # task, since the head is a standalone module count_parameters(
        # model._network, True) can't see (same visibility gap as
        # ridge_extra_param_count fixes for total_params).
        self._latest_head_trainable_params = 0
        # Two-ridge variant (use_task_ridge): per-task closed-form class
        # ridge classifiers. idx -> solved W_t [M, inc]; and a running
        # element count for ridge_extra_param_count (stored, non-trainable).
        self._task_ridge_cache = {}
        self._task_ridge_param_total = 0

        # Ridge classifiers (RanPAC-style decorrelation). Shared random
        # projection P + Gram/class-sum accumulators, held IN MEMORY for the
        # whole run so any previously routed task can still be scored.
        # Inference (ridge_scope "global_router"): the global ridge on the
        # frozen features routes top-1 to a task, then that task's
        # TOSCA-expert classifier classifies within the task. The expert
        # classifier is a small CosineLinear trained from scratch every task
        # on just that task's own classes (local labels) -- not a shared,
        # ever-growing fc, and not a second ridge system. Because it's never
        # touched again after its own task's training loop, there's no
        # cross-task interference to guard against; it's saved to disk once
        # (see _save_expert_head) and reloaded at inference by whichever
        # task the global router routes to.
        self._ridge_scope = str(self.args.get("ridge_scope", "global_router"))
        if (
            not self.args.get("use_ridge", False)
            or self._ridge_scope != "global_router"
        ):
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
        self._ridge_W_cache = {}  # ("__global__", lambda) -> [M, total_classes]
        # Global-scope running accumulators (shared frozen-feature space).
        self._ridge_G_global = None  # [M, M]
        self._ridge_C_global = None  # [M, total_classes]
        # router_label_space="task": literal Eq. 1-2 router -- a ridge fit
        # directly on task-label one-hots (T columns, growing by one per new
        # task) instead of RanPAC's class-label one-hots reduced to per-task
        # scores afterward (router_label_space="class", the default and what
        # ridge_scope="global_router" has always actually computed). G is
        # shared with the class-ridge accumulator (routing features only,
        # independent of which label target is used); only C and W differ.
        self._ridge_C_task_global = None  # [M, num_tasks]
        self._ridge_W_task_cache = {}  # ("__task__", lambda) -> [M, num_tasks]
        # Ablation only: learned task-gate router (router_type "linear"/"mlp").
        # A grown-per-task classifier over the projected features, trained
        # INCREMENTALLY on each task's own data -> so it forgets earlier
        # tasks, unlike the closed-form ridge router. None for the default
        # router_type "ridge".
        self._gate = None

    def incremental_train(self, data_manager):
        self._setup_task_loaders(data_manager)
        self._train()

    def _setup_task_loaders(self, data_manager):
        """Bookkeeping + dataloaders for the current task, split out from
        incremental_train so an offline script can replay task boundaries
        (task_ranges, test_loader) without running gradient training -- it
        loads saved adapters/expert heads/ridge state instead."""
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )

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
        # It should always run with suhuffle=Trueokay
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

    def _train(self):
        if bool(self.args.get("use_shared_head", False)):
            self._train_shared_head()
        else:
            self._train_independent_heads()

    def _train_independent_heads(self):
        task_train_start = time.perf_counter()
        self._network.to(self._device)
        self._set_trainable()

        # Fresh, independent classifier head for this task ONLY -- trained
        # from scratch on this task's own inc classes (local labels), never
        # touched again after this loop. No shared/growing fc, so no
        # cross-task gradient interference to guard against later.
        # Under the two-ridge variant (use_task_ridge) this head is used
        # ONLY to give the LuCA adapter a differentiable training signal;
        # the actual class decision at inference is made by a per-task
        # closed-form ridge fit in replace_fc (_fit_task_ridge), not by
        # this head.
        inc = self._classes_per_task()
        start = self._task_ranges[self._cur_task][0]
        head = CosineLinear(self._network.feature_dim, inc).to(self._device)

        optimizer = self.get_optimizer(
            lr=self.args["lr"], extra_params=head.parameters()
        )
        scheduler = self.get_scheduler(optimizer, self.args["epochs"])

        backbone_train_start = time.perf_counter()
        prog_bar = tqdm(range(self.args["epochs"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            head.train()
            losses = 0.0
            correct, total = 0, 0
            for _, inputs, targets in self.train_loader:
                inputs, targets = (
                    inputs.to(self._device),
                    targets.long().to(self._device),
                )
                local_targets = targets - start

                optimizer.zero_grad()
                feats = self._get_backbone()(inputs)
                logits = head(feats)["logits"]
                loss = F.cross_entropy(logits, local_targets)
                l1_loss = sum(
                    p.abs().sum() for p in self._network.backbone.tosca.parameters()
                )
                loss = loss + self.args["l1"] * l1_loss
                loss.backward()
                optimizer.step()

                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(local_targets.expand_as(preds)).cpu().sum()
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
        self._save_expert_head(head)
        if self._cur_task == 0 and bool(self.args.get("ffn_adapt", True)):
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

    def _train_shared_head(self):
        """use_shared_head experiment: ONE growing CosineLinear
        (self._network.fc) instead of Prism's usual per-task independent
        heads -- mirrors the original TOSCA paper's classifier. update_fc
        grows fc to total_classes, preserving already-learned rows and
        zero-initializing this task's new ones (utils/inc_net.py). Trained
        jointly with the TOSCA adapter via cross-entropy over ALL classes
        seen so far (global labels, not remapped to this task's local
        range), so -- like the original -- gradients touch older tasks'
        rows too; there is deliberately no independent-heads-style isolation
        here. Afterwards this task's own newly-added rows are overwritten
        with class-mean prototypes (_refresh_shared_head_prototypes),
        exactly like the original's replace_fc. The ridge router is
        unchanged: it still picks which task's TOSCA to load at inference,
        and only that task's own fc slice is trusted (see
        _routed_ridge_from_tasks) -- only the classifier head's internals
        match the original now, not the routing architecture around it."""
        task_train_start = time.perf_counter()
        self._network.to(self._device)
        self._network.update_fc(self._total_classes)
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
        if self._cur_task == 0 and bool(self.args.get("ffn_adapt", True)):
            self._save_adaptmlp()
        self._refresh_shared_head_prototypes()

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

    def _refresh_shared_head_prototypes(self):
        """use_shared_head experiment only. Overwrites just this task's
        newly-added fc rows with class-mean prototype embeddings computed
        under this task's just-trained TOSCA -- mirrors the original's
        replace_fc exactly, including reloading the just-saved TOSCA weights
        from disk rather than trusting the in-memory copy (matches the
        original's own redundant reload)."""
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
            self._network.fc.weight.data[class_index] = proto.to(self._device)

    def _get_backbone(self):
        if isinstance(self._network, torch.nn.DataParallel):
            return self._network.module.backbone
        return self._network.backbone

    def _set_trainable(self):
        # Tosca adapter trainable every task (reset per task, see
        # after_task). ViT's own adaptmlp trainable on task 0 only, frozen
        # afterwards. The per-task expert head is a standalone module (see
        # _train_independent_heads), not part of self._network, so it needs
        # no entry here.
        for p in self._network.parameters():
            p.requires_grad = False
        backbone = self._get_backbone()
        for p in backbone.tosca.parameters():
            p.requires_grad = True
        # ffn_adapt=False (frozen-vanilla-ViT ablation) builds no adaptmlp
        # params, so this loop is a no-op then; guard it anyway for clarity.
        if self._cur_task == 0 and bool(self.args.get("ffn_adapt", True)):
            for name, p in backbone.vit.named_parameters():
                if "adaptmlp" in name:
                    p.requires_grad = True
        # use_shared_head experiment: fc IS a registered self._network
        # submodule (unlike the per-task heads), so it needs explicitly
        # unfreezing here too.
        if (
            bool(self.args.get("use_shared_head", False))
            and self._network.fc is not None
        ):
            for p in self._network.fc.parameters():
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

    def _router_features(self, feats):
        """Feature space the GLOBAL ROUTER operates in. Ablation hook:
        router_space='raw' routes directly on the (L2-normalized) frozen
        [CLS] features with no projection; otherwise ('projected', default)
        it uses the RanPAC-style random-feature lift (_ridge_features). Used
        at both router fit (_accumulate_global_ridge) and router eval
        (_get_global_routed_ridge_logits) so the two stay in the same space.
        The per-task CLASS ridge/head is unaffected -- this only ablates the
        routing space."""
        if str(self.args.get("router_space", "projected")) == "raw":
            if bool(self.args.get("ridge_normalize", True)):
                feats = F.normalize(feats, p=2, dim=1)
            return feats
        return self._ridge_features(feats)

    # ----------------------------------------------------- learned gate (ablation)
    def _router_task_scores(self, vit_features, lam=None):
        """Per-task routing scores [B, num_tasks] according to router_type:
        'ridge' (default) uses the closed-form global ridge; 'linear'/'mlp'
        use the incrementally-trained learned gate. Only the ROUTER changes;
        the per-task class decision downstream is unaffected.

        Under router_type='ridge', router_label_space selects what the ridge
        is actually fit against: 'class' (default) fits RanPAC's own
        class-label ridge and reduces class logits to per-task scores via
        _task_scores_from_logits; 'task' fits a SEPARATE ridge directly on
        task-label one-hots (the literal Eq. 1-2 router), whose output is
        already [B, num_tasks] with no reduction step."""
        phi = self._router_features(vit_features)
        if str(self.args.get("router_type", "ridge")) == "ridge":
            if lam is None:
                lam = float(self.args.get("ridge_lambda", 1e4))
            if str(self.args.get("router_label_space", "class")) == "task":
                return phi @ self._global_task_ridge_weight(lam)
            return self._task_scores_from_logits(phi @ self._global_ridge_weight(lam))
        return self._gate(phi)  # [B, num_tasks]

    def _ensure_gate(self, num_tasks):
        """Create / grow the learned gate to num_tasks outputs, preserving
        already-learned rows. 'linear' = one Linear(M, T); 'mlp' = Linear(M,
        H)->ReLU->Linear(H, T)."""
        rt = str(self.args.get("router_type", "ridge"))
        M = int(self.args.get("ridge_proj_dim", 5000))
        H = int(self.args.get("gate_hidden", 512))
        dev = self._device
        if self._gate is None:
            if rt == "mlp":
                self._gate = nn.Sequential(
                    nn.Linear(M, H), nn.ReLU(), nn.Linear(H, num_tasks)
                ).to(dev)
            else:
                self._gate = nn.Linear(M, num_tasks).to(dev)
            return
        head = self._gate[-1] if isinstance(self._gate, nn.Sequential) else self._gate
        if head.out_features < num_tasks:
            grown = nn.Linear(head.in_features, num_tasks).to(dev)
            with torch.no_grad():
                grown.weight[: head.out_features] = head.weight
                grown.bias[: head.out_features] = head.bias
            if isinstance(self._gate, nn.Sequential):
                self._gate[-1] = grown
            else:
                self._gate = grown

    def _train_task_gate(self, phi, task_idx):
        """Incrementally train the learned gate on THIS task's projected
        features only, all labeled task_idx -- exactly the CIL setting (no old
        data), so the gate drifts toward the latest task and forgets earlier
        ones. This is the point of the learned-gate baseline; the closed-form
        ridge router avoids it by re-solving from accumulated statistics."""
        self._ensure_gate(self._cur_task + 1)
        dev = self._device
        phi = phi.to(dev).float()
        targets = torch.full((phi.size(0),), int(task_idx), dtype=torch.long, device=dev)
        opt = optim.Adam(self._gate.parameters(), lr=float(self.args.get("gate_lr", 1e-3)))
        loader = DataLoader(
            TensorDataset(phi, targets), batch_size=256, shuffle=True
        )
        self._gate.train()
        for _ in range(int(self.args.get("gate_epochs", 10))):
            for xb, yb in loader:
                opt.zero_grad()
                F.cross_entropy(self._gate(xb), yb).backward()
                opt.step()
        self._gate.eval()
        logging.info(
            "Learned gate (%s) trained on task %s: M=%d, N=%d.",
            self.args.get("router_type"),
            task_idx,
            phi.size(1),
            phi.size(0),
        )

    def _fit_task_ridge(self, adapted, labels):
        """Two-ridge variant (use_task_ridge): fit a per-task closed-form
        ridge CLASS classifier on this task's LuCA-adapted features, in the
        SAME random-projected M-dim space the global router uses. This is
        the second ridge: the global router ridge picks the task, then this
        per-task ridge picks the class within it. W_t = (Phi^T Phi + lam I)^-1
        Phi^T Y_local, with Phi = ridge_features(adapted) and Y_local the
        one-hot over this task's own inc classes. Stored, never gradient-
        trained, so it adds to total params but not trainable params."""
        lam = float(self.args.get("ridge_lambda", 1e4))
        adapted = adapted.to(self._device)
        start = self._task_ranges[self._cur_task][0]
        inc = self._classes_per_task()
        phi = self._ridge_features(adapted)  # [N, M], same projection P
        local = labels.to(self._device).long() - start
        onehot = torch.zeros(phi.size(0), inc, device=self._device)
        onehot[torch.arange(phi.size(0), device=self._device), local] = 1.0
        # float64 solve + in-place ridge regularization (see
        # _global_ridge_weight); G is a fresh temporary here, so mutating its
        # diagonal is safe and needs no clone.
        G = (phi.t() @ phi).double()  # [M, M] float64
        C = (phi.t() @ onehot).double()  # [M, inc] float64
        G.diagonal().add_(lam)
        W = torch.linalg.solve(G, C).float()  # [M, inc]
        self._task_ridge_cache[self._cur_task] = W
        self._task_ridge_param_total += W.numel()
        self._save_task_ridge(W)
        logging.info(
            "Task-ridge class classifier fit for task %s: M=%d, N=%d, classes=%d.",
            self._cur_task,
            phi.size(1),
            phi.size(0),
            inc,
        )

    def _save_task_ridge(self, W):
        path = os.path.join(self._ckpt_dir(), f"task_ridge{self._cur_task}.pth")
        torch.save({"W": W.cpu()}, path)

    def _load_task_ridge(self, idx):
        W = self._task_ridge_cache.get(idx)
        if W is None:
            state = torch.load(
                os.path.join(self._ckpt_dir(), f"task_ridge{idx}.pth"),
                map_location=self._device,
            )
            W = state["W"]
            self._task_ridge_cache[idx] = W
        return W

    def ridge_extra_param_count(self):
        """Element count of the classifier weights that live outside
        self._network (the global router's random projection P and solved
        head, plus every per-task expert head's params), so
        count_parameters(self._network) can be corrected to reflect the
        model's true total size. Each expert head is a standalone
        CosineLinear created fresh in _train and never registered as a
        submodule of self._network, so count_parameters(self._network)
        can't see it.
        """
        extra = 0
        if self._ridge_P is not None:
            extra += self._ridge_P.numel()
        if self._ridge_C_global is not None:
            extra += self._ridge_C_global.numel()
        extra += self._expert_param_total
        # Two-ridge variant: the per-task closed-form class ridges (W_t) are
        # stored, non-trainable classifier weights -- count them in total
        # params, not trainable params.
        extra += self._task_ridge_param_total
        return extra

    def expert_head_trainable_param_count(self):
        """Trainable param count of the most-recently-trained task's expert
        head -- see _latest_head_trainable_params. trainer.py adds this to
        count_parameters(model._network, True) so the logged
        'trainable_params' for a task includes the head that was actually
        being gradient-trained during it."""
        return self._latest_head_trainable_params

    def _save_expert_head(self, head):
        """Persist this task's classifier head -- a small CosineLinear
        trained from scratch on just this task's own inc classes (local
        labels, see _train). It's never touched again after this task's
        training loop finishes, so unlike a single shared/growing fc there's
        no cross-task gradient interference to guard against; this is a
        plain save, not a rescue. Loaded back at inference via
        _load_expert_head to classify within whichever task the global
        ridge router routes to."""
        t = self._cur_task
        state = {"weight": head.weight.detach().cpu()}
        if getattr(head, "sigma", None) is not None:
            state["sigma"] = head.sigma.detach().cpu()
        head_params = sum(p.numel() for p in head.parameters())
        self._expert_param_total += head_params
        self._latest_head_trainable_params = head_params
        path = os.path.join(self._ckpt_dir(), f"head_task{t}.pth")
        torch.save(state, path)
        logging.info("Expert head saved for task %s.", t)

    def _load_expert_head(self, idx):
        # head_task{idx}.pth is written once and never modified, so the
        # loaded state can be cached to avoid repeated disk reads (hot in
        # per-batch eval routing).
        state = self._expert_head_cache.get(idx)
        if state is None:
            state = torch.load(
                os.path.join(self._ckpt_dir(), f"head_task{idx}.pth"),
                map_location=self._device,
            )
            self._expert_head_cache[idx] = state
        return state

    def _routed_ridge_from_tasks(self, vit_features, topk_tasks):
        """Given shared ViT features and a per-sample set of candidate tasks
        [B, k] (from the global-ridge router), score each routed expert and
        scatter to the global class union. Normally (independent heads) each
        expert is its own independently-trained CosineLinear (see
        _train_independent_heads, _save_expert_head); under the
        use_shared_head experiment there is one growing self._network.fc
        instead, and only the routed task's own class-slice of it is
        trusted. Either way scores are cosine similarities scaled by sigma,
        so they remain cross-expert comparable and the argmax over the
        union is still a JOINT decision."""
        inc = self._classes_per_task()
        device = vit_features.device
        batch_size = vit_features.size(0)
        total_classes = self._total_classes
        starts = torch.tensor([s for s, _ in self._task_ranges], device=device)
        shared_head = bool(self.args.get("use_shared_head", False))

        out_logits = torch.full(
            (batch_size, total_classes), float("-inf"), device=device
        )
        for task_idx in torch.unique(topk_tasks).tolist():
            task_idx = int(task_idx)
            row_ids = (topk_tasks == task_idx).any(dim=1).nonzero(as_tuple=True)[0]
            self._load_tosca(task_idx)
            feats = self._get_backbone().forward_tosca(vit_features[row_ids])
            start = int(starts[task_idx].item())
            if bool(self.args.get("use_task_ridge", False)):
                # Two-ridge: class decision by the per-task closed-form ridge
                # on LuCA-adapted features in the router's random-feature space.
                W_task = self._load_task_ridge(task_idx).to(device)
                scores = self._ridge_features(feats) @ W_task
            elif shared_head:
                scores = self._network.fc(feats)["logits"][:, start : start + inc]
            else:
                head = self._load_expert_head(task_idx)
                weight = head["weight"].to(device)
                scores = F.linear(
                    F.normalize(feats, p=2, dim=1), F.normalize(weight, p=2, dim=1)
                )
                if "sigma" in head:
                    scores = head["sigma"].to(device) * scores
            out_logits[row_ids, start : start + inc] = scores

        return out_logits

    def _task_scores_from_logits(self, logits):
        """Class logits [B, C] -> per-task score [B, num_tasks]. Used to route
        from the global ridge's own class-level output.

        router_task_reduction (default 'max'): the routing decision is driven
        by whichever SINGLE class anywhere has the highest raw ridge score --
        one confidently-scored class in the wrong task's block can outvote
        every class in the sample's true task. 'sum' pools every class score
        within each task's block instead, so one outlier class is diluted by
        its (equal-count, since increments are uniform per dataset) siblings
        rather than deciding routing alone -- a hypothesis for why routing
        errors concentrate the way Group 1's ablation shows on ImageNet-A."""
        device = logits.device
        reduction = str(self.args.get("router_task_reduction", "max"))
        scores = torch.full(
            (logits.size(0), len(self._task_ranges)), float("-inf"), device=device
        )
        for t, (start, end) in enumerate(self._task_ranges):
            block = logits[:, start:end]
            scores[:, t] = block.sum(dim=1) if reduction == "sum" else block.max(dim=1).values
        return scores

    def _get_global_routed_ridge_logits(self, inputs, oracle_tasks=None, lam=None):
        """global_router: the GLOBAL ridge routes, then the per-task TOSCA
        expert -- a gradient-trained linear (cosine) classifier head --
        classifies within the chosen task. Top-1 routing -- validated best
        over k=1/2/5 (larger k only injects impostor experts)."""
        if lam is None:
            lam = float(self.args.get("ridge_lambda", 1e4))
        vit_features = self._extract_backbone_features(inputs)
        task_scores = self._router_task_scores(vit_features, lam)
        top1_task = task_scores.argmax(dim=1, keepdim=True)  # [B, 1]
        if oracle_tasks is not None:
            # Routing ceiling: send every sample to its GROUND-TRUTH task
            # expert, so the resulting accuracy isolates the experts' own
            # within-task discrimination from the router's mistakes.
            route_tasks = oracle_tasks.view(-1, 1)
        else:
            route_tasks = top1_task
        out_logits = self._routed_ridge_from_tasks(vit_features, route_tasks)
        return out_logits, top1_task.squeeze(1)

    def _accumulate_global_ridge(self, frozen, labels):
        """RanPAC-style single classifier: accumulate one shared Gram
        G = Phi^T Phi and class-sum C = Phi^T Y over ALL classes, in the frozen
        (pre-tosca) feature space. Decorrelates across tasks. Grows C's
        columns as new classes arrive."""
        total = self._total_classes
        frozen = frozen.to(self._device)
        labels = labels.to(self._device).long()
        phi = self._router_features(frozen)  # [N, router-space dim]
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

    def _accumulate_global_task_ridge(self, frozen):
        """router_label_space='task': accumulate the class-sum analogue for
        a ridge fit directly on task-label one-hots, i.e. Eq. 1-2's Y^task,
        rather than RanPAC's class-label one-hots. All rows passed in belong
        to self._cur_task (this is only ever called from replace_fc on the
        current task's own training data), so the one-hot is a single column
        of ones at index self._cur_task -- no per-sample label lookup needed.
        G is NOT recomputed here; it's shared with _accumulate_global_ridge
        since it depends only on the routing features, not the label target."""
        num_tasks = self._cur_task + 1
        phi = self._router_features(frozen.to(self._device))  # [N, M]
        onehot = torch.zeros(phi.size(0), num_tasks, device=self._device)
        onehot[:, self._cur_task] = 1.0
        C = (phi.t() @ onehot).float()
        if self._ridge_C_task_global is None:
            self._ridge_C_task_global = C
        else:
            prev = self._ridge_C_task_global
            grown = torch.zeros(prev.size(0), num_tasks, device=self._device)
            grown[:, : prev.size(1)] = prev
            self._ridge_C_task_global = grown + C
        self._ridge_W_task_cache.clear()
        logging.info(
            "Global task-label ridge accumulated Task %s: M=%d, N=%d, tasks=%d.",
            self._cur_task,
            phi.size(1),
            phi.size(0),
            num_tasks,
        )

    def _global_task_ridge_weight(self, lam):
        key = ("__task__", float(lam))
        W = self._ridge_W_task_cache.get(key)
        if W is None:
            A = self._ridge_G_global.double()
            A.diagonal().add_(float(lam))
            W = torch.linalg.solve(A, self._ridge_C_task_global.double()).float()
            self._ridge_W_task_cache[key] = W
        return W

    def _global_ridge_weight(self, lam):
        key = ("__global__", float(lam))
        W = self._ridge_W_cache.get(key)
        if W is None:
            # Solve in float64 for numerical precision, but add the ridge
            # term in place on a fresh float64 copy of G rather than
            # materializing a dense [M, M] identity and an [M, M] sum. This
            # is bit-identical to the old `solve(G.double() + lam*eye, ...)`
            # yet drops ~2x[M, M] of transient allocation, so the solve no
            # longer sets the training-memory peak.
            A = self._ridge_G_global.double()
            A.diagonal().add_(float(lam))
            W = torch.linalg.solve(A, self._ridge_C_global.double()).float()
            self._ridge_W_cache[key] = W
        return W

    def after_task(self):
        self._network.backbone.reset_tosca()
        self._known_classes = self._total_classes

    def replace_fc(self):
        """Fit the global router ridge from the frozen (pre-tosca) features
        of this task's own data -- exemplar-free, only Gram/class-sum stored,
        never raw samples. Under the two-ridge variant (use_task_ridge) also
        fit the per-task closed-form CLASS ridge on the LuCA-adapted features
        of the same data (_fit_task_ridge); the throwaway gradient head from
        _train only shaped the adapter and is not used at inference. The
        router ridge uses FROZEN features (task decision), the class ridge
        uses LuCA-ADAPTED features (class decision), both through the same
        random projection."""
        use_task_ridge = bool(self.args.get("use_task_ridge", False))
        self._network.eval()
        frozen_list = []
        adapted_list = []
        label_list = []
        with torch.no_grad():
            for _, data, label in self.train_loader_for_protonet:
                data = data.to(self._device)
                label = label.long().to(self._device)
                vit = self._extract_backbone_features(data)
                frozen_list.append(vit.cpu())
                if use_task_ridge:
                    adapted = self._get_backbone().forward_tosca(vit)
                    adapted_list.append(adapted.cpu())
                label_list.append(label.cpu())
        label_list = torch.cat(label_list, dim=0)
        frozen = torch.cat(frozen_list, dim=0)
        if str(self.args.get("router_type", "ridge")) == "ridge":
            self._accumulate_global_ridge(frozen, label_list)
            if str(self.args.get("router_label_space", "class")) == "task":
                self._accumulate_global_task_ridge(frozen)
        else:
            # Learned-gate router (ablation): train the gate on this task's
            # projected features -> task index (incremental; forgets old tasks).
            with torch.no_grad():
                phi = self._router_features(frozen.to(self._device))
            self._train_task_gate(phi, self._cur_task)
        if use_task_ridge:
            self._fit_task_ridge(torch.cat(adapted_list, dim=0), label_list)

    def get_optimizer(self, lr, extra_params=None):
        params = list(filter(lambda p: p.requires_grad, self._network.parameters()))
        if extra_params is not None:
            params += list(extra_params)
        if self.args["optimizer"] == "sgd":
            optimizer = optim.SGD(
                params,
                momentum=0.9,
                lr=lr,
                weight_decay=self.args["weight_decay"],
            )
        elif self.args["optimizer"] == "adam":
            optimizer = optim.Adam(
                params,
                lr=lr,
                weight_decay=self.args["weight_decay"],
            )
        elif self.args["optimizer"] == "adamw":
            optimizer = optim.AdamW(
                params,
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
        if getattr(self, "_last_routing_acc", None) is not None:
            metrics["routing_acc"] = self._last_routing_acc
            logging.info(
                "Routing accuracy (top-1 task): {:.3f}".format(self._last_routing_acc)
            )
        # Oracle ceiling: re-evaluate the SAME trained experts but route with
        # ground-truth task ids, isolating accuracy lost purely to routing.
        if bool(self.args.get("log_oracle", False)):
            yo_pred, yo_true = self._eval_cnn(self.test_loader, oracle=True)
            oracle_accy = self._evaluate(yo_pred, yo_true)
            metrics["oracle_top1"] = oracle_accy["top1"]
            logging.info(
                "Oracle accuracy (ground-truth routing, top-1): {:.3f}".format(
                    oracle_accy["top1"]
                )
            )
        return cnn_accy, nme_accy, metrics

    def _true_task_from_targets(self, targets):
        true_task = torch.zeros_like(targets)
        for t, (start, end) in enumerate(self._task_ranges):
            true_task[(targets >= start) & (targets < end)] = t
        return true_task

    def _eval_cnn(self, loader, oracle=False):
        self._network.eval()
        y_pred, y_true = [], []
        route_correct, route_total = 0, 0
        log_routing = bool(self.args.get("log_routing_acc", False))

        for _, (_, inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(self._device), targets.long().to(self._device)
            with torch.no_grad():
                oracle_tasks = self._true_task_from_targets(targets) if oracle else None
                outputs, top1_task = self._get_global_routed_ridge_logits(
                    inputs, oracle_tasks=oracle_tasks
                )
                if log_routing:
                    true_task = self._true_task_from_targets(targets)
                    route_correct += (top1_task == true_task).sum().item()
                    route_total += targets.size(0)
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        self._last_routing_acc = (
            100.0 * route_correct / route_total if route_total else None
        )
        return np.concatenate(y_pred), np.concatenate(y_true)

    def _ckpt_dir(self):
        """Namespace all of Prism's persisted checkpoints (tosca adapter /
        adaptmlp / ridge) by dataset + prefix + seed so concurrent or
        sequential runs -- different datasets, hyperparameter grid variants,
        or the 5-seed benchmark runs of the SAME dataset -- never read/write
        each other's files. Without the seed tag, seed 1994 would silently
        overwrite seed 1993's saved adapters/ridge matrices, breaking offline
        sweeps and resumes."""
        dataset = str(self.args.get("dataset", "data"))
        prefix = re.sub(
            r"[^A-Za-z0-9_.-]+", "_", str(self.args.get("prefix", "")).strip()
        )
        tag = f"{dataset}__{prefix}" if prefix else dataset
        seed = self.args.get("seed")
        if isinstance(seed, (int, np.integer)):
            tag = f"{tag}__seed{int(seed)}"
        path = os.path.join("prism", tag)
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
        """Persist the SOLVED global ridge weight W (not the raw G/C Gram
        matrix) for reproducibility/inspection -- a few KB total, vs. G being
        a dense [M, M] matrix (~900MB at M=15000). The snapshot uses a single
        fixed filename since only the latest one is ever needed once training
        has moved past a task. The per-task expert classifier (a standalone
        CosineLinear) is saved separately by _save_expert_head, right after
        training, before replace_fc runs."""
        t = self._cur_task
        ckpt_dir = self._ckpt_dir()
        lam = float(self.args.get("ridge_lambda", 1e4))
        if self._ridge_G_global is not None:
            torch.save(
                {"W": self._global_ridge_weight(lam).cpu(), "lambda": lam},
                os.path.join(ckpt_dir, "ridge_global.pth"),
            )
        logging.info("Global ridge state saved for task %s.", t)

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

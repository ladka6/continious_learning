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
        # parameters, for extra_param_count() -- these heads are
        # standalone CosineLinear modules, never registered inside
        # self._network, so count_parameters(self._network) can't see them.
        self._expert_param_total = 0
        # Trainable param count of the CURRENT task's expert head at the
        # moment its training loop finished (see _save_expert_head) --
        # exposed via expert_head_trainable_param_count() so trainer.py can
        # correct the "trainable_params" metric it logs right after this
        # task, since the head is a standalone module count_parameters(
        # model._network, True) can't see (same visibility gap as
        # extra_param_count fixes for total_params).
        self._latest_head_trainable_params = 0

        # Router: a growing CosineLinear classifier over a FIXED random
        # projection of frozen (pre-tosca) ViT features -- same random-
        # feature expansion the old ridge router used (RanPAC-style: makes
        # classes more linearly separable), but the classifier on top is now
        # gradient-trained, not a closed-form solve. It grows by one block
        # of `inc` rows per task -- each block is a small CosineLinear
        # trained ONLY during its own task (see _train), then detached and
        # frozen into _router_weight_blocks/_router_sigma_blocks forever
        # after. Because each block is a SEPARATE tensor concatenated at
        # scoring time (not one shared growing nn.Parameter), earlier blocks
        # are structurally immune to gradient updates from later tasks -- no
        # cross-task interference to guard against, unlike a single shared
        # fc kept trainable across all tasks. Cosine (not plain linear)
        # specifically because independently-trained blocks have no shared
        # constraint on output scale otherwise -- a plain linear router
        # collapsed almost all routing to whichever block had the largest
        # weight norm, confirmed empirically before this was cosine-scored.
        self._router_P = None  # [feat_dim, M] fixed random projection
        self._router_weight_blocks = []  # [inc, M] per task
        self._router_sigma_blocks = []  # scalar per task

    def incremental_train(self, data_manager):
        self._setup_task_loaders(data_manager)
        self._train()

    def _setup_task_loaders(self, data_manager):
        """Bookkeeping + dataloaders for the current task, split out from
        incremental_train so an offline script can replay task boundaries
        (task_ranges, test_loader) without running gradient training -- it
        loads saved adapters/expert heads/router blocks instead."""
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
        self.test_loader = DataLoader(
            test_dataset, batch_size=48, shuffle=False, num_workers=num_workers
        )

    def _train(self):
        task_train_start = time.perf_counter()
        self._network.to(self._device)
        self._set_trainable()

        # Fresh, independent classifier head for this task ONLY -- trained
        # from scratch on this task's own inc classes (local labels), never
        # touched again after this loop. No shared/growing fc, so no
        # cross-task gradient interference to guard against later.
        inc = self._classes_per_task()
        start = self._task_ranges[self._cur_task][0]
        head = CosineLinear(self._network.feature_dim, inc).to(self._device)

        # Router block for this task ONLY -- a CosineLinear (not a plain
        # linear layer) over the FIXED random-projected expansion of frozen
        # (pre-tosca) ViT features (see _router_features), trained on local
        # labels just like the head. CosineLinear specifically because each
        # task's block is trained completely independently (its own local
        # softmax, never sees other tasks' classes) -- a plain linear layer
        # has no constraint tying its output SCALE to any other block's, so
        # whichever block ends up with the largest weight norm silently
        # dominates every joint argmax at routing time regardless of the
        # input (verified empirically: one block's score stayed frozen
        # across 6 straight evals while every other block scored exactly
        # zero). Cosine similarity bounds every block's output to [-1, 1]
        # regardless of weight norm, keeping independently-trained blocks
        # comparable -- the same reason this already works for the expert
        # heads. Frozen and appended to _router_weight_blocks/
        # _router_sigma_blocks after this loop; previous blocks are never
        # touched here since they're not part of this module at all.
        router_new = CosineLinear(
            int(self.args.get("ridge_proj_dim", 5000)), inc
        ).to(self._device)

        optimizer = self.get_optimizer(
            lr=self.args["lr"], extra_params=head.parameters()
        )
        # Fully separate optimizer for the router, so its backward/step is
        # entirely decoupled from the adapter+head's -- eliminates any
        # possibility of the router's loss (an independently, from-scratch
        # trained CosineLinear on a 15000-dim input) interfering with
        # adapter/head training, which empirically collapsed when the two
        # were combined into one shared loss.backward() call.
        router_optimizer = self.get_optimizer(
            lr=self.args["lr"], extra_params=router_new.parameters()
        )
        scheduler = self.get_scheduler(optimizer, self.args["epochs"])
        router_scheduler = self.get_scheduler(router_optimizer, self.args["epochs"])

        backbone_train_start = time.perf_counter()
        prog_bar = tqdm(range(self.args["epochs"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            head.train()
            router_new.train()
            losses = 0.0
            correct, router_correct, total = 0, 0, 0
            for _, inputs, targets in self.train_loader:
                inputs, targets = (
                    inputs.to(self._device),
                    targets.long().to(self._device),
                )
                local_targets = targets - start

                optimizer.zero_grad()
                backbone = self._get_backbone()
                vit_feats = backbone.forward_features(inputs)
                tosca_feats = backbone.forward_tosca(vit_feats)
                logits = head(tosca_feats)["logits"]
                loss = F.cross_entropy(logits, local_targets)
                l1_loss = sum(
                    p.abs().sum() for p in self._network.backbone.tosca.parameters()
                )
                loss = loss + self.args["l1"] * l1_loss
                loss.backward()
                optimizer.step()

                router_optimizer.zero_grad()
                router_logits = router_new(
                    self._router_features(vit_feats.detach())
                )["logits"]
                router_loss = F.cross_entropy(router_logits, local_targets)
                router_loss.backward()
                router_optimizer.step()

                losses += loss.item() + router_loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(local_targets.expand_as(preds)).cpu().sum()
                _, router_preds = torch.max(router_logits, dim=1)
                router_correct += (
                    router_preds.eq(local_targets.expand_as(router_preds)).cpu().sum()
                )
                total += len(targets)

            if scheduler is not None:
                scheduler.step()
            if router_scheduler is not None:
                router_scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            router_acc = np.around(
                tensor2numpy(router_correct) * 100 / total, decimals=2
            )

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Router_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args["epochs"],
                losses / len(self.train_loader),
                train_acc,
                router_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)
        backbone_train_seconds = time.perf_counter() - backbone_train_start
        self._save_tosca()
        self._save_expert_head(head)
        self._save_router_block(router_new)
        if self._cur_task == 0:
            self._save_adaptmlp()

        task_train_seconds = time.perf_counter() - task_train_start

        self._latest_task_timing = {
            "backbone_train_seconds": backbone_train_seconds,
            "replace_fc_seconds": 0.0,
            "task_train_seconds": task_train_seconds,
        }
        logging.info(
            "Task {} timing => backbone_train {:.2f}s, total_train {:.2f}s".format(
                self._cur_task,
                backbone_train_seconds,
                task_train_seconds,
            )
        )

    def _get_backbone(self):
        if isinstance(self._network, torch.nn.DataParallel):
            return self._network.module.backbone
        return self._network.backbone

    def _set_trainable(self):
        # Tosca adapter trainable every task (reset per task, see
        # after_task). ViT's own adaptmlp trainable on task 0 only, frozen
        # afterwards. The per-task expert head is a standalone module (see
        # _train), not part of self._network, so it needs no entry here.
        for p in self._network.parameters():
            p.requires_grad = False
        backbone = self._get_backbone()
        for p in backbone.tosca.parameters():
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

    # ---------------------------------------------------------------- router
    def _save_router_block(self, router_new):
        """Freeze this task's router block (weight+sigma) and append it to
        _router_weight_blocks/_router_sigma_blocks -- it's never touched
        again after this point, so earlier tasks' routing decisions can't
        drift as later tasks train. Also persisted to disk for offline
        reproducibility, mirroring _save_expert_head."""
        t = self._cur_task
        weight = router_new.weight.detach().clone()
        sigma = router_new.sigma.detach().clone()
        self._router_weight_blocks.append(weight)
        self._router_sigma_blocks.append(sigma)
        torch.save(
            {"weight": weight.cpu(), "sigma": sigma.cpu()},
            os.path.join(self._ckpt_dir(), f"router_task{t}.pth"),
        )
        logging.info("Router block saved for task %s.", t)

    def _router_projection(self, feature_dim):
        """Fixed shared random projection [feat_dim -> M] (RanPAC-style
        lift) feeding the router's linear classifier. Created once, seeded
        for reproducibility."""
        if self._router_P is None:
            M = int(self.args.get("ridge_proj_dim", 5000))
            gen = torch.Generator(device="cpu").manual_seed(1993)
            P = torch.randn(feature_dim, M, generator=gen)
            self._router_P = P.to(self._device)
        return self._router_P

    def _router_features(self, feats):
        """L2-normalize (optional), project, ReLU -> [N, M]."""
        if bool(self.args.get("ridge_normalize", True)):
            feats = F.normalize(feats, p=2, dim=1)
        phi = feats @ self._router_projection(feats.size(1))
        if str(self.args.get("ridge_activation", "relu")) == "relu":
            phi = F.relu(phi)
        return phi

    def _router_logits(self, vit_features):
        """Score the random-projected expansion of frozen ViT features
        against every router block trained so far. Each block scores its
        own classes via cosine similarity (normalized input/weight) scaled
        by that block's OWN sigma, THEN the per-block scores are
        concatenated -- this per-block cosine scoring is what keeps
        independently-trained blocks comparable at the joint argmax (see
        __init__); concatenating raw per-block logits before any
        normalization would reintroduce the scale-domination problem."""
        feats = F.normalize(self._router_features(vit_features), p=2, dim=1)
        scores = [
            sigma * F.linear(feats, F.normalize(weight, p=2, dim=1))
            for weight, sigma in zip(
                self._router_weight_blocks, self._router_sigma_blocks
            )
        ]
        return torch.cat(scores, dim=1)

    def extra_param_count(self):
        """Element count of the classifier weights that live outside
        self._network (the router's fixed random projection P, every router
        block's weight+sigma, plus every per-task expert head's params), so
        count_parameters(self._network) can be corrected to reflect the
        model's true total size. All are standalone tensors/modules created
        outside self._network, so count_parameters(self._network) can't
        see them.
        """
        extra = 0
        if self._router_P is not None:
            extra += self._router_P.numel()
        extra += sum(w.numel() for w in self._router_weight_blocks)
        extra += sum(s.numel() for s in self._router_sigma_blocks)
        extra += self._expert_param_total
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
        _load_expert_head to classify within whichever task the router
        routes to."""
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

    def _score_routed_experts(self, vit_features, topk_tasks, oracle_tasks=None):
        """Given shared ViT features and a per-sample set of candidate tasks
        [B, k] (from the router), score each routed expert with
        its own independently-trained CosineLinear head (see _train,
        _save_expert_head) and scatter to the global class union. Scores are
        cosine similarities scaled by that task's own sigma, so they remain
        cross-expert comparable and the argmax over the union is still a
        JOINT decision."""
        inc = self._classes_per_task()
        device = vit_features.device
        batch_size = vit_features.size(0)
        total_classes = self._total_classes
        starts = torch.tensor([s for s, _ in self._task_ranges], device=device)
        offsets = torch.arange(inc, device=device)

        out_logits = torch.full(
            (batch_size, total_classes), float("-inf"), device=device
        )
        for task_idx in torch.unique(topk_tasks).tolist():
            task_idx = int(task_idx)
            row_ids = (topk_tasks == task_idx).any(dim=1).nonzero(as_tuple=True)[0]
            self._load_tosca(task_idx)
            feats = self._get_backbone().forward_tosca(vit_features[row_ids])
            head = self._load_expert_head(task_idx)
            weight = head["weight"].to(device)
            scores = F.linear(
                F.normalize(feats, p=2, dim=1), F.normalize(weight, p=2, dim=1)
            )
            if "sigma" in head:
                scores = head["sigma"].to(device) * scores
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
        each task's block). Used to route from the router's own output."""
        device = logits.device
        scores = torch.full(
            (logits.size(0), len(self._task_ranges)), float("-inf"), device=device
        )
        for t, (start, end) in enumerate(self._task_ranges):
            scores[:, t] = logits[:, start:end].max(dim=1).values
        return scores

    def _get_routed_logits(self, inputs, oracle_tasks=None):
        """The router (growing linear classifier over frozen ViT features,
        see _router_logits) routes top-1, then the per-task TOSCA expert --
        a gradient-trained linear (cosine) classifier head -- classifies
        within the chosen task. Top-1 routing -- validated best over
        k=1/2/5 (larger k only injects impostor experts) back when this was
        ridge-based; assumed to still hold, not re-validated for the linear
        router."""
        vit_features = self._extract_backbone_features(inputs)
        global_logits = self._router_logits(vit_features)
        task_scores = self._task_scores_from_logits(global_logits)
        top1_task = task_scores.argmax(dim=1, keepdim=True)  # [B, 1]
        return self._score_routed_experts(vit_features, top1_task, oracle_tasks)

    def after_task(self):
        self._network.backbone.reset_tosca()
        self._known_classes = self._total_classes

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

        # Router: normalize + random projection (768->M) + ReLU + linear
        # head to all seen classes -- classifier is gradient-trained now,
        # but the feature expansion is unchanged. (The per-task expert cost
        # is separate.)
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
                outputs = self._get_routed_logits(
                    inputs, oracle_tasks=oracle_tasks
                )
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)

    def _ckpt_dir(self):
        """Namespace all persisted checkpoints (tosca/adaptmlp/router/head)
        by dataset + prefix + seed so concurrent or sequential runs --
        different datasets, hyperparameter grid variants, or the 5-seed
        benchmark runs of the SAME dataset -- never read/write each other's
        files. Without the seed tag, seed 1994 would silently overwrite seed
        1993's saved adapters/blocks, breaking offline sweeps and resumes."""
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

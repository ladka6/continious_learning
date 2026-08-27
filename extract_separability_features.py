"""Dump raw and projected [CLS] features for Figure `separability` (Observation 2):
task separability before/after the random-feature lift, on ImageNet-A.

Reuses the PRISM Learner's own feature paths (_extract_backbone_features,
_router_features) so the "projected" features are byte-for-byte the same
space the actual router uses -- normalize -> random projection -> ReLU,
with the same fixed M=15000, seed=1993 projection matrix used everywhere
else in the paper (see models/prism.py::_ridge_projection: the projection
seed is hardcoded and independent of the run's own --seed, so it is
identical across every PRISM run regardless of which checkpoint/seed this
script loads).

This does NOT train anything: it replays the config's task boundaries via
_setup_task_loaders (the offline-replay path _setup_task_loaders was split
out for, see its docstring in prism.py) and loads the already-trained
first-session AdaptMLP checkpoint from disk. No TOSCA per-task adapters are
needed here, since Observation 2 is specifically about the FROZEN backbone
representation, before any task-specific adaptation.

Usage (run on Snellius, from continious_learning/):
    python extract_separability_features.py \
        --config exps/ablation/router_relu15k.json --seed 1993 \
        --n-per-task 150 --out separability_features.npz
"""
import argparse
import copy
import json

import numpy as np
import torch

from utils import factory
from utils.data_manager import DataManager


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="exps/ablation/router_relu15k.json")
    ap.add_argument("--seed", type=int, default=1993)
    ap.add_argument("--n-per-task", type=int, default=150)
    ap.add_argument("--out", default="separability_features.npz")
    cli = ap.parse_args()

    with open(cli.config) as f:
        args = json.load(f)
    args["seed"] = cli.seed
    args["device"] = ["0" if torch.cuda.is_available() else "cpu"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_manager = DataManager(
        args["dataset"], args["shuffle"], args["seed"],
        args["init_cls"], args["increment"], args,
    )
    args["nb_classes"] = data_manager.nb_classes
    args["nb_tasks"] = data_manager.nb_tasks

    model = factory.get_model(args["model_name"], args)
    model._device = device
    model._network.to(device)

    # Offline replay: walk every task boundary without training, so
    # _task_ranges/_known_classes/_total_classes end up exactly as they were
    # after the real training run finished, then reload its AdaptMLP.
    # after_task() must run each iteration too -- _known_classes only
    # advances there, not in _setup_task_loaders, so skipping it would leave
    # _known_classes stuck at 0 and every task_ranges entry identically
    # (0, task_size) instead of the real cumulative boundaries. reset_tosca()
    # inside after_task() is harmless here since this script never touches
    # TOSCA-adapted features, only the frozen backbone.
    for _ in range(data_manager.nb_tasks):
        model._setup_task_loaders(data_manager)
        model.after_task()
    model._load_adaptmlp()
    model._network.eval()

    raw_feats, proj_feats, task_labels = [], [], []
    with torch.no_grad():
        for _, inputs, targets in model.test_loader:
            inputs = inputs.to(device)
            targets = targets.long()
            feats = model._extract_backbone_features(inputs)  # frozen [CLS], [B, 768]
            proj = model._router_features(feats)  # normalize -> project -> ReLU, [B, M]
            true_task = model._true_task_from_targets(targets.to(device))
            raw_feats.append(feats.cpu())
            proj_feats.append(proj.cpu().half())  # half precision: M=15000 is large
            task_labels.append(true_task.cpu())

    raw_feats = torch.cat(raw_feats, dim=0).numpy()
    proj_feats = torch.cat(proj_feats, dim=0).numpy()
    task_labels = torch.cat(task_labels, dim=0).numpy()

    # Subsample n_per_task per task for a readable, tractable UMAP plot.
    rng = np.random.default_rng(cli.seed)
    keep = []
    for t in np.unique(task_labels):
        idx = np.where(task_labels == t)[0]
        rng.shuffle(idx)
        keep.append(idx[: cli.n_per_task])
    keep = np.concatenate(keep)

    np.savez_compressed(
        cli.out,
        raw_feats=raw_feats[keep],
        proj_feats=proj_feats[keep],
        task_labels=task_labels[keep],
    )
    print(f"Wrote {cli.out}: {len(keep)} samples, "
          f"raw={raw_feats[keep].shape}, proj={proj_feats[keep].shape}, "
          f"{len(np.unique(task_labels))} tasks")


if __name__ == "__main__":
    main()

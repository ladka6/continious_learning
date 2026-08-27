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

Extracts from each task's TRAINING data (train_loader_for_protonet), the
same loader replace_fc() uses to fit the real router, not the small
held-out test set: a first version of this script used ~150 test
samples/task and its UMAP plot showed almost no raw-vs-projected
separation gap even though the extraction itself was verified correct (a
linear probe on the raw features matched the published 37.4% almost
exactly) -- because a ridge classifier fit on only ~150 points/task in a
15000-dim projected space is nowhere near the well-conditioned regime the
real router benefits from when fit on the full accumulated training set.
Using the training data directly closes that regime gap.

Also dumps per-sample CLASS labels (0..total_classes-1), not just task
labels, so a separate script can replicate the REAL router mechanism
exactly -- a 200-way class-level ridge reduced to 10 task scores via
per-task max-pooling (_accumulate_global_ridge + _task_scores_from_logits)
-- instead of the weaker direct-task-classifier proxy every earlier probe
in this investigation used, which is why those all underestimated the
projected space's real advantage.

This does NOT train anything: it replays the config's task boundaries via
_setup_task_loaders (the offline-replay path _setup_task_loaders was split
out for, see its docstring in prism.py) and loads the already-trained
first-session AdaptMLP checkpoint from disk. No TOSCA per-task adapters are
needed here, since Observation 2 is specifically about the FROZEN backbone
representation, before any task-specific adaptation.

Usage (run on Snellius, from continious_learning/):
    python extract_separability_features.py \
        --config exps/ablation/router_relu15k.json --seed 1993 \
        --n-per-task 500 --out separability_features.npz
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
    ap.add_argument("--n-per-task", type=int, default=500)
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

    # _ckpt_dir() (used by _load_adaptmlp) keys only on dataset/prefix/seed,
    # not on task state, so this can run before or after the replay loop;
    # doing it first mirrors the real run's order (AdaptMLP is trained once,
    # during task 0, before any later task runs).
    model._load_adaptmlp()
    model._network.eval()

    raw_feats, proj_feats, task_labels, class_labels = [], [], [], []
    with torch.no_grad():
        for _ in range(data_manager.nb_tasks):
            # _setup_task_loaders rebuilds train_loader_for_protonet for
            # JUST this task's own new classes (mode="test" transforms on
            # the training split) -- the exact loader replace_fc() uses to
            # fit the real router, task by task, as training proceeds.
            model._setup_task_loaders(data_manager)
            cur_task = model._cur_task
            for _, inputs, targets in model.train_loader_for_protonet:
                inputs = inputs.to(device)
                feats = model._extract_backbone_features(inputs)  # frozen [CLS], [B, 768]
                proj = model._router_features(feats)  # normalize -> project -> ReLU, [B, M]
                raw_feats.append(feats.cpu())
                proj_feats.append(proj.cpu().half())  # half precision: M=15000 is large
                task_labels.append(torch.full((inputs.size(0),), cur_task, dtype=torch.long))
                class_labels.append(targets.long())  # already GLOBAL class ids, 0..199
            # after_task() must still run each iteration -- _known_classes
            # only advances there, not in _setup_task_loaders, and the NEXT
            # iteration's task-boundary bookkeeping depends on it. Its
            # reset_tosca() is harmless here since this script never touches
            # TOSCA-adapted features, only the frozen backbone.
            model.after_task()

    raw_feats = torch.cat(raw_feats, dim=0).numpy()
    proj_feats = torch.cat(proj_feats, dim=0).numpy()
    task_labels = torch.cat(task_labels, dim=0).numpy()
    class_labels = torch.cat(class_labels, dim=0).numpy()

    # Subsample n_per_task per task for a readable, tractable plot.
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
        class_labels=class_labels[keep],
        total_classes=np.array([args["nb_classes"]]),
        classes_per_task=np.array([args["increment"]]),
    )
    print(f"Wrote {cli.out}: {len(keep)} samples, "
          f"raw={raw_feats[keep].shape}, proj={proj_feats[keep].shape}, "
          f"{len(np.unique(task_labels))} tasks, "
          f"{len(np.unique(class_labels))} classes")


if __name__ == "__main__":
    main()

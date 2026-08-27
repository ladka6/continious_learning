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

Third iteration. History, each one motivating the next:
  v1: ~150 held-out TEST samples/task -> UMAP showed almost no raw-vs-
      projected gap. A raw-feature linear probe matched the published
      37.4% almost exactly, so extraction wasn't the problem -- 150
      points/task in a 15000-dim space is nowhere near the regime the
      real router (fit on the full training set) benefits from.
  v2: switched to each task's full TRAINING pool (~750 img/task, still
      capped at 500/task) -- closed some of the gap in a quick probe
      (42%->53%) but a *direct 10-way task classifier* (which is what
      UMAP/a probe/LDA all effectively fit) is fundamentally a weaker
      design than the REAL router: a 200-way CLASS-level ridge reduced to
      10 task scores via per-task max-pooling. That gap doesn't close with
      more data because it's a different mechanism, not a data problem.
  v3 (this version): replicates the real mechanism exactly by fitting on
      the FULL per-task training pool (uncapped) and evaluating on the
      REAL cumulative TEST set, matching the exact train/fit vs.
      test/evaluate split the paper's own numbers come from -- unlike v2's
      quick reproduction, which fit AND evaluated on a random split of the
      same capped training pool (no real distribution shift between
      "train" and "test" there, which inflated raw's apparent accuracy and
      collapsed the true gap).

Saves two separate pools: TRAIN (full, for fitting the ridge) and TEST
(the real cumulative test set, for evaluating + visualizing), each with
both raw and projected features plus per-sample class AND task labels.

This does NOT train anything: it replays the config's task boundaries via
_setup_task_loaders (the offline-replay path _setup_task_loaders was split
out for, see its docstring in prism.py) and loads the already-trained
first-session AdaptMLP checkpoint from disk. No TOSCA per-task adapters are
needed here, since Observation 2 is specifically about the FROZEN backbone
representation, before any task-specific adaptation.

Usage (run on Snellius, from continious_learning/):
    python extract_separability_features.py \
        --config exps/ablation/router_relu15k.json --seed 1993 \
        --out separability_features_full.npz
"""
import argparse
import json

import numpy as np
import torch

from utils import factory
from utils.data_manager import DataManager


def _extract(loader, model, device):
    raw_feats, proj_feats, class_labels = [], [], []
    with torch.no_grad():
        for _, inputs, targets in loader:
            inputs = inputs.to(device)
            feats = model._extract_backbone_features(inputs)  # frozen [CLS], [B, 768]
            proj = model._router_features(feats)  # normalize -> project -> ReLU, [B, M]
            raw_feats.append(feats.cpu())
            proj_feats.append(proj.cpu().half())  # half precision: M=15000 is large
            class_labels.append(targets.long())  # already GLOBAL class ids
    return (
        torch.cat(raw_feats, dim=0),
        torch.cat(proj_feats, dim=0),
        torch.cat(class_labels, dim=0),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="exps/ablation/router_relu15k.json")
    ap.add_argument("--seed", type=int, default=1993)
    ap.add_argument("--out", default="separability_features_full.npz")
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

    train_raw, train_proj, train_class, train_task = [], [], [], []
    for _ in range(data_manager.nb_tasks):
        # _setup_task_loaders rebuilds train_loader_for_protonet for JUST
        # this task's own new classes (mode="test" transforms on the
        # training split) -- the exact loader replace_fc() uses to fit the
        # real router, task by task, as training proceeds. FULL pool, no
        # subsampling, to match the real router's actual fitting data.
        model._setup_task_loaders(data_manager)
        cur_task = model._cur_task
        raw, proj, cls = _extract(model.train_loader_for_protonet, model, device)
        train_raw.append(raw)
        train_proj.append(proj)
        train_class.append(cls)
        train_task.append(torch.full((raw.size(0),), cur_task, dtype=torch.long))
        # after_task() must still run each iteration -- _known_classes only
        # advances there, not in _setup_task_loaders, and the NEXT
        # iteration's task-boundary bookkeeping depends on it. Its
        # reset_tosca() is harmless here since this script never touches
        # TOSCA-adapted features, only the frozen backbone.
        model.after_task()

    train_raw = torch.cat(train_raw, dim=0).numpy()
    train_proj = torch.cat(train_proj, dim=0).numpy()
    train_class = torch.cat(train_class, dim=0).numpy()
    train_task = torch.cat(train_task, dim=0).numpy()

    # model.test_loader now holds the REAL cumulative test set (all 200
    # classes, official test images) -- exactly what eval_task() scores the
    # published 37.43%/63.86% routing accuracy against.
    test_raw, test_proj, test_class = _extract(model.test_loader, model, device)
    test_raw = test_raw.numpy()
    test_proj = test_proj.numpy()
    test_class = test_class.numpy()
    classes_per_task = int(args["increment"])
    test_task = (test_class // classes_per_task).astype(np.int64)

    np.savez_compressed(
        cli.out,
        train_raw_feats=train_raw, train_proj_feats=train_proj,
        train_class_labels=train_class, train_task_labels=train_task,
        test_raw_feats=test_raw, test_proj_feats=test_proj,
        test_class_labels=test_class, test_task_labels=test_task,
        total_classes=np.array([args["nb_classes"]]),
        classes_per_task=np.array([classes_per_task]),
    )
    print(f"Wrote {cli.out}: train={train_raw.shape[0]} samples "
          f"({len(np.unique(train_task))} tasks), "
          f"test={test_raw.shape[0]} samples ({len(np.unique(test_task))} tasks)")


if __name__ == "__main__":
    main()

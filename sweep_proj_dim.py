"""Offline ridge-projection-dim (M) sweep -- Tier 1: forward passes only, no
gradient training. Reuses replace_fc() exactly as production training does
(prototypes + ridge fit from real forward passes over each task's training
images), just with a fresh Learner (and fresh random P) per candidate M.
Needs the saved TOSCA adapters + AdaptMLP from a completed (or in-progress,
up to the target task) *_ridge_router.json run -- adapter training itself
does not depend on M, so it is not repeated here.

Cost: for each M, one forward-only pass over every task's training set up to
the target task (~1/20th the cost of that many training epochs, since there's
no backward pass). Does NOT overwrite the run's saved tosca/ridge_*.pth files
-- nothing here is persisted, this is eval-only.

Usage:
    python sweep_proj_dim.py --config exps/tosca_ina_ridge_router.json \
        --proj-dims 2000,5000,10000 --task 2 --lambda 1000
"""
import argparse
import json

from utils import factory
from utils.data_manager import DataManager
from trainer import _set_random, _set_device


def load_json(path):
    with open(path) as f:
        return json.load(f)


def run_one(base_args, data_manager, target_task, proj_dim, lam):
    args = dict(base_args)
    args["ridge_proj_dim"] = proj_dim
    model = factory.get_model(args["model_name"], args)
    model._network.to(model._device)
    if lam is not None:
        model.args["ridge_lambda"] = lam

    for t in range(target_task + 1):
        model._setup_task_loaders(data_manager)
        if t == 0:
            model._load_adaptmlp()
        model.replace_fc()  # forward-only: prototypes + ridge fit, no backprop
        if t < target_task:
            model._known_classes = model._total_classes

    y_pred, y_true = model._eval_cnn(model.test_loader)
    return 100.0 * (y_pred[:, 0] == y_true).sum() / len(y_true)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--proj-dims", type=str, default="2000,5000,10000",
        help="comma-separated M values (random projection output dim) to try",
    )
    parser.add_argument(
        "--task", type=int, default=-1,
        help="task index to evaluate at (0-based, -1 = last task)",
    )
    parser.add_argument(
        "--lambda", dest="lam", type=float, default=None,
        help="override ridge_lambda for all M values (defaults to config's)",
    )
    cli = parser.parse_args()

    base_args = load_json(cli.config)
    seed = base_args["seed"][0]
    # Match the trainer: Learner._ckpt_dir tags checkpoints with the scalar seed.
    base_args["seed"] = seed
    _set_random(seed)
    _set_device(base_args)

    assert base_args.get("use_ridge") and base_args.get("ridge_scope") == "global_router", (
        "sweep_proj_dim.py only supports ridge_scope=global_router configs."
    )

    data_manager = DataManager(
        base_args["dataset"], base_args["shuffle"], seed,
        base_args["init_cls"], base_args["increment"], base_args,
    )
    base_args["nb_classes"] = data_manager.nb_classes
    base_args["nb_tasks"] = data_manager.nb_tasks

    target_task = cli.task if cli.task >= 0 else data_manager.nb_tasks - 1
    assert 0 <= target_task < data_manager.nb_tasks

    proj_dims = [int(x) for x in cli.proj_dims.split(",")]
    print(f"Task {target_task}, lambda={cli.lam if cli.lam is not None else base_args['ridge_lambda']} "
          f"-- proj_dim sweep (forward passes only, no retraining):\n")
    results = []
    for M in proj_dims:
        acc = run_one(base_args, data_manager, target_task, M, cli.lam)
        results.append((M, acc))
        print(f"  M={M}: top1={acc:.2f}")

    best_M, best_acc = max(results, key=lambda r: r[1])
    print(f"\nBest: M={best_M} -> {best_acc:.2f}")


if __name__ == "__main__":
    main()

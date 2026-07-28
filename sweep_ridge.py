"""Offline ridge-lambda sweep: reload a finished run's saved adapters and
ridge matrices (G, C) from disk and try several lambda values WITHOUT
retraining -- W=(G+lam*I)^-1 C is a fresh closed-form solve each time, no
gradient step, no re-reading training images. Minutes, not hours.

Requires a run of the corresponding *_ridge_router.json config to have
completed first (or reached the target task), so tosca/*.pth and
tosca/ridge_*.pth exist. Must run where that tosca/ directory lives (same
cwd main.py was run from).

Usage:
    python sweep_ridge.py --config exps/tosca_ina_ridge_router.json \
        --lambdas 10,100,1000,10000,100000 --task -1
"""
import argparse
import json

from utils import factory
from utils.data_manager import DataManager
from trainer import _set_random, _set_device


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--lambdas", type=str, default="10,100,1000,10000,100000",
        help="comma-separated lambda values to try",
    )
    parser.add_argument(
        "--task", type=int, default=-1,
        help="task index to evaluate at (0-based, -1 = last task)",
    )
    cli = parser.parse_args()

    args = load_json(cli.config)
    seed = args["seed"][0]
    # Learner._ckpt_dir tags checkpoints with the scalar seed; match the
    # trainer, which sets args["seed"] = seed before building the model.
    args["seed"] = seed
    _set_random(seed)
    _set_device(args)

    assert args.get("use_ridge") and args.get("ridge_scope") == "global_router", (
        "sweep_ridge.py only supports ridge_scope=global_router configs "
        "(the mode currently in use)."
    )

    data_manager = DataManager(
        args["dataset"], args["shuffle"], seed, args["init_cls"], args["increment"], args
    )
    args["nb_classes"] = data_manager.nb_classes
    args["nb_tasks"] = data_manager.nb_tasks

    target_task = cli.task if cli.task >= 0 else data_manager.nb_tasks - 1
    assert 0 <= target_task < data_manager.nb_tasks

    model = factory.get_model(args["model_name"], args)

    print(f"Replaying task bookkeeping 0..{target_task} and loading saved state "
          f"(no training, no gradient steps)...")
    for t in range(target_task + 1):
        model._setup_task_loaders(data_manager)
        if t == 0:
            model._load_adaptmlp()
        model._load_tosca(t)
        model._load_ridge_task(t)
        model._load_ridge_global(t)
        if t < target_task:
            model._known_classes = model._total_classes

    model._network.to(model._device)
    model._network.eval()

    lambdas = [float(x) for x in cli.lambdas.split(",")]
    print(f"\nTask {target_task} ({model._known_classes if target_task==0 else model._task_ranges[target_task][1]} "
          f"classes seen) -- lambda sweep, top-1 global-ridge routing:\n")
    results = []
    for lam in lambdas:
        model.args["ridge_lambda"] = lam
        model._ridge_W_cache.clear()
        y_pred, y_true = model._eval_cnn(model.test_loader)
        acc = 100.0 * (y_pred[:, 0] == y_true).sum() / len(y_true)
        results.append((lam, acc))
        print(f"  lambda={lam:g}: top1={acc:.2f}")

    best_lam, best_acc = max(results, key=lambda r: r[1])
    print(f"\nBest: lambda={best_lam:g} -> {best_acc:.2f}")


if __name__ == "__main__":
    main()

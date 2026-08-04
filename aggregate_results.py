"""Aggregate benchmark results (ours + PILOT baselines) into mean +/- std
tables over the 5 seeds (1993-1997).

Both trainers write one ``*_metrics.json`` per (config, seed) with per-task
accuracy and efficiency metrics; this script walks any number of log roots,
groups runs by (model, dataset), and emits:

  results/accuracy.csv      final top-1 and average incremental acc, mean/std
  results/efficiency.csv    train time, eval time, memory, FLOPs, params, ...
  stdout                    the same, human-readable

Training FLOPs are ESTIMATED as 3 x inference_flops_per_sample x
train_samples x epochs per task (fwd+bwd ~ 3x fwd) for methods that train
every task. Methods in TRAIN_ONLY_TASK0 (ranpac, aper_adapter) only run a
gradient-based training loop on task 0 -- later tasks are closed-form
(ridge/prototype), doing a single forward pass over the task's training
samples for feature extraction, not `epochs` many forward+backward passes.
Charging every task at the full epoch count (as an earlier version of this
script did) overestimated these methods' training cost by roughly the
number of tasks (~20x on a 20-task dataset) -- verified directly against
models/ranpac.py and models/aper_adapter.py in the PILOT repo, both of
which literally skip training past task 0
(`if self._cur_task == 0: ...train... else: pass`).

FIXED (was a known gap): TOSCA's own total_params used to exclude its ridge
classifier weights (the M=15000-dim projection and solved per-task/
global-router heads), since they're stored as plain attributes on the
Learner rather than inside self._network's nn.Parameters -- unlike RanPAC,
whose equivalent solved ridge weight lives inside self._network.fc.weight
and IS counted by count_parameters(). trainer.py now adds
model.ridge_extra_param_count() (models/tosca.py) on top of
count_parameters(model._network) for total_params, computed from the
persisted per-task/global _ridge_C accumulators (not the lazily-populated
_ridge_W_cache, which may not cover every seen task). Only affects
total_params, not trainable_params -- these weights are closed-form
solved, never requires_grad=True, same as RanPAC's equivalent. Runs logged
before this fix (any run without the corrected total_params) will still
show the old, undercounted total_params_m -- rerun to get the corrected
figure.

Wall-clock train time is the measured ground truth next to any of the above.

Usage:
    python aggregate_results.py --roots logs ../pilot_baselines/logs
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

# Methods whose gradient-based training only happens on task 0; later tasks
# are closed-form (ridge/prototype), not `epochs`-many forward+backward
# passes. See module docstring.
TRAIN_ONLY_TASK0 = {"ranpac", "aper_adapter"}

# EASE's config uses init_epochs/later_epochs (both 20 in exps/ease.json),
# not epochs/tuned_epoch, so MetricsLogger's meta capture
# (args.get("epochs", args.get("tuned_epoch"))) never finds a value and
# train_gflops_est comes back blank -- even though EASE genuinely trains
# every task. Both keys share the same value here, so this override is
# exact, not a guess.
EPOCHS_OVERRIDE = {"ease": 20}

# EASE and MOS keep one adapter per task and ENSEMBLE across all of them at
# eval time -- inference_flops_per_sample genuinely grows ~linearly with the
# number of tasks seen (verified: ease_cifar224's final-task value / 20
# tasks ~ 35.4 GFLOPs, matching the base ViT-B/16 cost seen for every other
# method). That's real and worth reporting as-is for the Inference GFLOPs
# column. But during TRAINING these methods only forward through the
# CURRENT task's single new adapter, not the full accumulated ensemble --
# using each task's own (increasingly ensemble-inflated) inference cost as
# the per-sample training-forward proxy overestimates later tasks' training
# FLOPs by ~n_tasks_so_far x, compounding across the run (this produced
# ease_cifar224's ~1.1 BILLION GFLOPs estimate against a measured 1.9h
# wall-clock train time). Use task 0's inference cost -- measured before any
# ensembling has accumulated -- as a constant single-adapter proxy instead.
ENSEMBLE_EVAL_METHODS = {"ease", "mos"}

# Dedicated reduced-epoch, single-seed runs (config flag profile_train_flops)
# that measure REAL forward+backward FLOPs via torch.profiler instead of the
# analytic estimate above. Filename prefix "profileflops_" keeps them
# entirely separate from the accuracy-bearing 5-seed runs -- they use far
# fewer epochs, so their own accuracy/timing numbers are meaningless and
# must never be mixed into the accuracy or seed-averaged efficiency tables.
PROFILE_PREFIX = "profileflops_"


def load_profiled_flops(roots):
    """-> {(model, dataset): [task_dicts]} from dedicated profiling runs.
    Used only to replace the analytic train_gflops_est with a measured
    value where available; never contributes to accuracy numbers."""
    profiled = {}
    for root in roots:
        for path in glob.glob(os.path.join(root, "**", "*_metrics.json"), recursive=True):
            if not os.path.basename(path).startswith(PROFILE_PREFIX):
                continue
            try:
                with open(path) as f:
                    run = json.load(f)
            except (OSError, json.JSONDecodeError):
                print(f"skipping unreadable {path}")
                continue
            meta, tasks = run.get("meta", {}), run.get("tasks", [])
            if not tasks:
                continue
            key = (str(meta.get("model_name")), str(meta.get("dataset")))
            profiled[key] = tasks
    return profiled


def load_runs(roots):
    """-> {(model, dataset): {seed: run_dict}}"""
    runs = defaultdict(dict)
    for root in roots:
        for path in glob.glob(os.path.join(root, "**", "*_metrics.json"), recursive=True):
            # PILOT baselines now report eval_shuffle=true (prefix "benchshuf")
            # as the canonical protocol; skip the old eval_shuffle=false runs
            # (prefix "bench") so they don't get mixed in or silently picked
            # over the shuffled ones for the same (model, dataset, seed).
            basename = os.path.basename(path)
            if basename.startswith("bench_") or basename.startswith(PROFILE_PREFIX):
                continue
            try:
                with open(path) as f:
                    run = json.load(f)
            except (OSError, json.JSONDecodeError):
                print(f"skipping unreadable {path}")
                continue
            meta, tasks = run.get("meta", {}), run.get("tasks", [])
            if not tasks:
                continue
            key = (str(meta.get("model_name")), str(meta.get("dataset")))
            seed = meta.get("seed")
            prev = runs[key].get(seed)
            # Keep the run that got furthest (re-runs / partial crashes).
            if prev is None or len(tasks) > len(prev["tasks"]):
                runs[key][seed] = {"meta": meta, "tasks": tasks, "path": path}
    return runs


def per_seed_summary(run, profiled_tasks=None):
    meta, tasks = run["meta"], run["tasks"]
    model_name = str(meta.get("model_name"))
    curve = [t.get("cnn_top1") for t in tasks if t.get("cnn_top1") is not None]
    epochs = meta.get("epochs") or EPOCHS_OVERRIDE.get(model_name) or 0

    if profiled_tasks and len(profiled_tasks) >= len(tasks):
        # Real torch.profiler-measured FLOPs available for this (model,
        # dataset) from a dedicated reduced-epoch profiling run. Scale each
        # task's measured FLOPs by (this run's real epoch count / the
        # profiling run's reduced epoch count) -- tasks that measured near
        # zero (e.g. adapt-once methods' later, non-training tasks) stay
        # correctly near zero regardless of the scale factor, since the
        # measurement itself (not an assumption) determined that.
        train_flops_est = 0.0
        have_flops = True
        for i in range(len(tasks)):
            pt = profiled_tasks[i]
            measured = pt.get("train_flops_measured")
            profiled_epochs = pt.get("profiled_epochs")
            if measured is None:
                have_flops = False
                continue
            scale = (epochs / profiled_epochs) if (epochs and profiled_epochs) else 1.0
            train_flops_est += measured * scale
    else:
        train_only_task0 = model_name in TRAIN_ONLY_TASK0
        ensemble_eval = model_name in ENSEMBLE_EVAL_METHODS
        base_f = tasks[0].get("inference_flops_per_sample") if ensemble_eval else None
        train_flops_est = 0.0
        have_flops = True
        for i, t in enumerate(tasks):
            f, n = t.get("inference_flops_per_sample"), t.get("train_samples")
            if ensemble_eval and base_f:
                f = base_f
            if not (f and n):
                have_flops = False
                continue
            if train_only_task0 and i > 0:
                # Closed-form task: one forward pass over the task's training
                # samples for feature extraction, no backward, no epoch loop.
                train_flops_est += 1.0 * f * n
            elif epochs:
                train_flops_est += 3.0 * f * n * epochs
            else:
                have_flops = False
    last = tasks[-1]
    return {
        "n_tasks": len(tasks),
        "final_top1": curve[-1] if curve else None,
        "avg_inc_acc": float(np.mean(curve)) if curve else None,
        "train_seconds_total": sum(t.get("train_seconds") or 0 for t in tasks),
        "eval_ms_per_sample": last.get("eval_ms_per_sample"),
        "inference_gflops_per_sample": (
            (last.get("inference_flops_per_sample") or 0) / 1e9 or None
        ),
        "train_gflops_est": train_flops_est / 1e9 if have_flops else None,
        "train_peak_mem_mb": max((t.get("train_peak_mem_mb") or 0) for t in tasks),
        "eval_peak_mem_mb": max((t.get("eval_peak_mem_mb") or 0) for t in tasks),
        "total_params_m": (last.get("total_params") or 0) / 1e6,
        "trainable_params_m": (last.get("trainable_params") or 0) / 1e6,
        "checkpoint_mb": last.get("checkpoint_mb"),
    }


METRICS = [
    "final_top1",
    "avg_inc_acc",
    "train_seconds_total",
    "eval_ms_per_sample",
    "inference_gflops_per_sample",
    "train_gflops_est",
    "train_peak_mem_mb",
    "eval_peak_mem_mb",
    "total_params_m",
    "trainable_params_m",
    "checkpoint_mb",
]


# Total classes / increment per dataset, used to catch runs truncated by a
# SLURM timeout. Comparing seeds against each other's max task count (the
# old heuristic) misses this: if all 5 seeds get killed by the same
# wall-clock limit, they're all truncated to the same task and look
# mutually "complete" even though none of them reached the real final task.
EXPECTED_TASKS = {
    "cifar224": 100 // 5,
    "cub": 200 // 10,
    "imageneta": 200 // 20,
    "imagenetr": 200 // 20,
    "omnibenchmark": 300 // 30,
    "vtab": 50 // 10,
}


def aggregate(runs, profiled=None):
    profiled = profiled or {}
    rows = []
    for (model, dataset), by_seed in sorted(runs.items()):
        profiled_tasks = profiled.get((model, dataset))
        summaries = [per_seed_summary(r, profiled_tasks) for r in by_seed.values()]
        expected_tasks = EXPECTED_TASKS.get(dataset)
        if expected_tasks is None:
            expected_tasks = max(s["n_tasks"] for s in summaries)
        complete = [s for s in summaries if s["n_tasks"] == expected_tasks]
        row = {
            "model": model,
            "dataset": dataset,
            "seeds": len(complete),
            "seeds_partial": len(summaries) - len(complete),
        }
        for m in METRICS:
            vals = [s[m] for s in complete if s.get(m) is not None]
            row[f"{m}_mean"] = float(np.mean(vals)) if vals else None
            row[f"{m}_std"] = float(np.std(vals)) if vals else None
        rows.append(row)
    return rows


def write_csv(rows, path, columns):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c) for c in columns})
    print(f"wrote {path}")


def fmt(mean, std):
    if mean is None:
        return "-"
    return f"{mean:.2f}±{std:.2f}" if std is not None else f"{mean:.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--roots", nargs="+", default=["logs", "../pilot_baselines/logs"],
        help="log roots to scan for *_metrics.json",
    )
    parser.add_argument("--out", type=str, default="results")
    cli = parser.parse_args()

    valid_roots = [r for r in cli.roots if os.path.isdir(r)]
    runs = load_runs(valid_roots)
    if not runs:
        print("No *_metrics.json found under: " + ", ".join(cli.roots))
        return
    profiled = load_profiled_flops(valid_roots)
    if profiled:
        print(f"Using measured train_gflops_est for {len(profiled)} (model, dataset) "
              f"pair(s) with dedicated profiling runs.")
    rows = aggregate(runs, profiled)
    os.makedirs(cli.out, exist_ok=True)

    acc_cols = ["model", "dataset", "seeds", "seeds_partial"] + [
        f"{m}_{s}" for m in ("final_top1", "avg_inc_acc") for s in ("mean", "std")
    ]
    eff_cols = ["model", "dataset", "seeds"] + [
        f"{m}_{s}" for m in METRICS[2:] for s in ("mean", "std")
    ]
    write_csv(rows, os.path.join(cli.out, "accuracy.csv"), acc_cols)
    write_csv(rows, os.path.join(cli.out, "efficiency.csv"), eff_cols)

    print("\n=== Accuracy (mean±std over seeds) ===")
    print(f"{'model':<16}{'dataset':<16}{'seeds':<7}{'final top1':<16}{'avg inc acc':<16}")
    for r in rows:
        print(
            f"{r['model']:<16}{r['dataset']:<16}{r['seeds']:<7}"
            f"{fmt(r['final_top1_mean'], r['final_top1_std']):<16}"
            f"{fmt(r['avg_inc_acc_mean'], r['avg_inc_acc_std']):<16}"
            + (f"  [{r['seeds_partial']} partial]" if r["seeds_partial"] else "")
        )

    print("\n=== Efficiency (mean±std over seeds) ===")
    hdr = ["model", "dataset", "train s", "eval ms/smp", "inf GF/smp",
           "train GF est", "peak mem MB", "params M", "ckpt MB"]
    print("".join(f"{h:<15}" for h in hdr))
    for r in rows:
        cells = [
            r["model"][:14], r["dataset"][:14],
            fmt(r["train_seconds_total_mean"], r["train_seconds_total_std"]),
            fmt(r["eval_ms_per_sample_mean"], r["eval_ms_per_sample_std"]),
            fmt(r["inference_gflops_per_sample_mean"], r["inference_gflops_per_sample_std"]),
            fmt(r["train_gflops_est_mean"], r["train_gflops_est_std"]),
            fmt(r["train_peak_mem_mb_mean"], r["train_peak_mem_mb_std"]),
            fmt(r["total_params_m_mean"], r["total_params_m_std"]),
            fmt(r["checkpoint_mb_mean"], r["checkpoint_mb_std"]),
        ]
        print("".join(f"{c:<15}" for c in cells))


if __name__ == "__main__":
    main()

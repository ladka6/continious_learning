"""Aggregate benchmark results (ours + PILOT baselines) into mean +/- std
tables over the 5 seeds (1993-1997).

Both trainers write one ``*_metrics.json`` per (config, seed) with per-task
accuracy and efficiency metrics; this script walks any number of log roots,
groups runs by (model, dataset), and emits:

  results/accuracy.csv      final top-1 and average incremental acc, mean/std
  results/efficiency.csv    train time, eval time, memory, FLOPs, params, ...
  stdout                    the same, human-readable

Training FLOPs are ESTIMATED as 3 x inference_flops_per_sample x
train_samples x epochs per task (fwd+bwd ~ 3x fwd) -- uniform across all
methods, since instrumenting every method's inner train loop is not viable.
Wall-clock train time is the measured ground truth next to it.

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


def load_runs(roots):
    """-> {(model, dataset): {seed: run_dict}}"""
    runs = defaultdict(dict)
    for root in roots:
        for path in glob.glob(os.path.join(root, "**", "*_metrics.json"), recursive=True):
            # PILOT baselines now report eval_shuffle=true (prefix "benchshuf")
            # as the canonical protocol; skip the old eval_shuffle=false runs
            # (prefix "bench") so they don't get mixed in or silently picked
            # over the shuffled ones for the same (model, dataset, seed).
            if os.path.basename(path).startswith("bench_"):
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


def per_seed_summary(run):
    meta, tasks = run["meta"], run["tasks"]
    curve = [t.get("cnn_top1") for t in tasks if t.get("cnn_top1") is not None]
    epochs = meta.get("epochs") or 0
    train_flops_est = 0.0
    have_flops = True
    for t in tasks:
        f, n = t.get("inference_flops_per_sample"), t.get("train_samples")
        if f and n and epochs:
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


def aggregate(runs):
    rows = []
    for (model, dataset), by_seed in sorted(runs.items()):
        summaries = [per_seed_summary(r) for r in by_seed.values()]
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

    runs = load_runs([r for r in cli.roots if os.path.isdir(r)])
    if not runs:
        print("No *_metrics.json found under: " + ", ".join(cli.roots))
        return
    rows = aggregate(runs)
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

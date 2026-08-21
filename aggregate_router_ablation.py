"""Aggregate the router-space ablation (Experiment A / C) over its seeds.

Reads the per-variant SLURM logs written by submit_router_ablation.sh
(logs/prism-abl-<variant>-*_*.out) and reports, per router variant:
  - Route acc : top-1 task-routing accuracy after the final stage
                (logged by models/prism.py when log_routing_acc=true)
  - Final acc : A_T, last-stage top-1 over all classes
  - Avg inc   : A_bar, mean over stages
mean +/- std across seeds (population std, matching aggregate_results.py).

Usage:  python aggregate_router_ablation.py
"""
import glob
import re
import numpy as np

VARIANTS = [
    ("raw", "Raw [CLS] (no projection)"),
    ("lin15k", "Proj linear, M=15000"),
    ("relu1k", "Proj + ReLU, M=1000"),
    ("relu5k", "Proj + ReLU, M=5000"),
    ("relu15k", "Proj + ReLU, M=15000"),
    ("relu25k", "Proj + ReLU, M=25000"),
]


def _ms(x):
    if not x:
        return "     n/a"
    a = np.array(x, dtype=float)
    return f"{a.mean():5.2f} ± {a.std():4.2f}"


def main():
    print(f"{'Router variant':28} {'Route acc':>14} {'Final acc':>14} {'Avg inc':>14}")
    print("-" * 74)
    for tag, label in VARIANTS:
        racc, fin, avg = [], [], []
        for f in sorted(glob.glob(f"logs/prism-abl-{tag}-*_*.out")):
            text = open(f).read()
            curves = [l for l in text.splitlines() if "CNN top1 curve" in l]
            if not curves:
                continue
            s = curves[-1].split("curve:")[1].replace("np.float64", "")
            nums = [float(x) for x in re.findall(r"[-+]?\d+\.\d+|\d+", s)]
            if not nums:
                continue
            fin.append(nums[-1])
            avg.append(float(np.mean(nums)))
            r = re.findall(r"Routing accuracy \(top-1 task\): ([\d.]+)", text)
            if r:
                racc.append(float(r[-1]))
        n = len(fin)
        print(
            f"{label:28} {_ms(racc):>14} {_ms(fin):>14} {_ms(avg):>14}   (n={n})"
        )


if __name__ == "__main__":
    main()

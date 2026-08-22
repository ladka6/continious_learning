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
    # Group 1: routing space + M-sweep (closed-form ridge router)
    ("raw", "Raw [CLS] (no projection)"),
    ("lin15k", "Proj linear (no ReLU), M=15000"),
    ("relu5k", "Proj + ReLU ridge, M=5000"),
    ("relu10k", "Proj + ReLU ridge, M=10000 (RanPAC)"),
    ("relu15k", "Proj + ReLU ridge, M=15000 (ours)"),
    ("relu20k", "Proj + ReLU ridge, M=20000"),
    # Group 2: learned gates (incremental), M=15000
    ("gate_lin", "Learned linear gate (no ReLU)"),
    ("gate_linrelu", "Learned linear gate (ReLU proj)"),
    ("gate_mlp", "Learned MLP gate (ReLU proj)"),
    # Group 3: ceiling / backbone controls (M=15000, ReLU ridge router)
    ("oracle", "Oracle routing (ground-truth)"),
    ("noadaptmlp", "Frozen ViT (no AdaptMLP) + LuCA"),
]


def _ms(x):
    if not x:
        return "     n/a"
    a = np.array(x, dtype=float)
    return f"{a.mean():5.2f} ± {a.std():4.2f}"


def main():
    print(
        f"{'Router variant':32} {'Route acc':>14} {'Final acc':>14} "
        f"{'Avg inc':>14} {'Oracle acc':>14}"
    )
    print("-" * 92)
    for tag, label in VARIANTS:
        # Keep only the most recent SLURM array (highest job id) per variant,
        # so a bugfix rerun's logs are not averaged together with the stale
        # ones it replaces (they share the prism-abl-<tag>-* prefix).
        jobs = {}
        for f in glob.glob(f"logs/prism-abl-{tag}-*_*.out"):
            m = re.search(rf"prism-abl-{re.escape(tag)}-(\d+)_\d+\.out$", f)
            if m:
                jobs.setdefault(int(m.group(1)), []).append(f)
        files = sorted(jobs[max(jobs)]) if jobs else []
        racc, fin, avg, orc = [], [], [], []
        for f in files:
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
            o = re.findall(
                r"Oracle accuracy \(ground-truth routing, top-1\): ([\d.]+)", text
            )
            if o:
                orc.append(float(o[-1]))
        n = len(fin)
        print(
            f"{label:32} {_ms(racc):>14} {_ms(fin):>14} {_ms(avg):>14} "
            f"{_ms(orc):>14}   (n={n})"
        )


if __name__ == "__main__":
    main()

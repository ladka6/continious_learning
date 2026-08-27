"""Render figs/fig_router_forgetting.pdf from the JSON dumped by
extract_router_forgetting_curve.py --json-out (run on Snellius, see that
script's docstring). Pull the JSON back locally first, e.g.:

    scp snellius:~/continious_learning/router_forgetting.json .

Usage:
    pip install matplotlib
    python make_router_forgetting_figure.py --in router_forgetting.json \
        --out paper/figs/fig_router_forgetting.pdf
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STYLE = {
    "ridge": dict(color="tab:blue", label="Closed-form ridge router"),
    "gate": dict(color="tab:red", label="Gradient-trained gate"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="router_forgetting.json")
    ap.add_argument("--out", default="paper/figs/fig_router_forgetting.pdf")
    args = ap.parse_args()

    with open(args.inp) as f:
        data = json.load(f)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for key, style in STYLE.items():
        if key not in data:
            continue
        mean = np.array(data[key]["mean"])
        std = np.array(data[key]["std"])
        stages = np.arange(1, len(mean) + 1)
        ax.plot(stages, mean, color=style["color"], label=style["label"], linewidth=2,
                marker="o", markersize=5)
        ax.fill_between(stages, mean - std, mean + std, color=style["color"], alpha=0.2)

    n_tasks = max(len(data[k]["mean"]) for k in data)
    ax.axhline(100.0 / n_tasks, color="gray", linestyle="--", linewidth=1,
               label=f"Chance ($1/t$, $t={n_tasks}$)")
    ax.set_xlabel("Incremental stage")
    ax.set_ylabel("Top-1 task-routing accuracy [%]")
    ax.set_xticks(np.arange(1, n_tasks + 1))
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

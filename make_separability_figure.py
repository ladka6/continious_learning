"""Build Figure `separability` (Observation 2) from real extracted features.

Consumes separability_features.npz produced by
extract_separability_features.py (run on Snellius, see run_extract_separability.sh),
pulled back locally e.g.:

    scp snellius:~/continious_learning/separability_features.npz .

Runs UMAP on the raw and projected features with identical hyperparameters
and random seed (UMAP is for visualization only; the quantitative claim in
the caption is the router's own task-routing accuracy, already measured in
Table `router-ablation`: 37.43% raw vs. 63.86% projected+ReLU, M=15000 --
not recomputed here, just annotated).

Usage:
    pip install umap-learn matplotlib
    python make_separability_figure.py --in separability_features.npz \
        --out paper/figs/fig_umap_mock.pdf
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import umap

RAW_ACC = 37.43
PROJ_ACC = 63.86


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="separability_features.npz")
    ap.add_argument("--out", default="paper/figs/fig_umap_mock.pdf")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = np.load(args.inp)
    raw_feats = data["raw_feats"].astype(np.float32)
    proj_feats = data["proj_feats"].astype(np.float32)
    task_labels = data["task_labels"]
    n_tasks = len(np.unique(task_labels))

    reducer_kwargs = dict(n_neighbors=30, min_dist=0.1, metric="cosine", random_state=args.seed)
    raw_emb = umap.UMAP(**reducer_kwargs).fit_transform(raw_feats)
    proj_emb = umap.UMAP(**reducer_kwargs).fit_transform(proj_feats)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    cmap = plt.get_cmap("tab10" if n_tasks <= 10 else "tab20")

    for ax, emb, title, acc in [
        (axes[0], raw_emb, "(a) Raw [CLS] features", RAW_ACC),
        (axes[1], proj_emb, "(b) Projected + ReLU features", PROJ_ACC),
    ]:
        for t in range(n_tasks):
            mask = task_labels == t
            ax.scatter(emb[mask, 0], emb[mask, 1], s=6, alpha=0.7,
                       color=cmap(t % cmap.N), label=f"T{t+1}")
        ax.set_title(f"{title}\nrouting acc: {acc:.1f}%", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=n_tasks, fontsize=7,
               bbox_to_anchor=(0.5, -0.05), frameon=False)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

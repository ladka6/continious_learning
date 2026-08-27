"""Build Figure `separability` (Observation 2) via a supervised LDA
projection instead of UMAP.

Context: a first version used UMAP (unsupervised, then weakly-supervised)
on the same extracted features and it never showed much more than a
modest raw-vs-projected gap, even after fixing a real extraction bug and
re-extracting from the full per-task TRAINING set (matching the regime the
real router is fit in). The reason turned out to be structural, not a
tuning problem: UMAP embeds the raw INPUT geometry, but the published
37.4%/63.9% routing-accuracy gap (Table `router-ablation`) comes from
fitting a 200-way CLASS-level ridge and reducing it to 10 task scores via
per-task max-pooling -- a supervised mechanism that lives in a fitted
classifier's weights, not in the input vectors' unsupervised geometry. No
amount of UMAP tuning surfaces a decision rule that only exists after
fitting that specific classifier.

LDA is a more honest tool for this specific claim: it explicitly finds the
directions that best separate the given task labels via between/within
scatter, i.e. it visualizes "the linearly discriminative structure
available to a supervised classifier", not restricted to one specific
2D layout. It is still not IDENTICAL to the real 200-column class-ridge +
max-pool mechanism (LDA here separates by task-label directly, with at
most n_tasks-1 discriminant directions), so the routing-accuracy numbers
annotated in each panel remain the actual quantitative claim, exactly as
before, not something this plot re-derives.

Usage:
    python make_separability_figure_lda.py --in separability_features.npz \
        --out paper/figs/fig_separability_lda.pdf
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

RAW_ACC = 37.43
PROJ_ACC = 63.86


def lda_embed(X, y, pca_dim=None):
    """2D LDA projection. For very high-dimensional X (proj_feats, M=15000)
    with far fewer samples than features, the within-class scatter matrix
    is singular; PCA to pca_dim first (a standard PCA+LDA pipeline) avoids
    that without changing what LDA is optimizing for."""
    if pca_dim is not None and X.shape[1] > pca_dim:
        X = PCA(n_components=pca_dim, random_state=0).fit_transform(X)
    return LinearDiscriminantAnalysis(n_components=2).fit_transform(X, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="separability_features.npz")
    ap.add_argument("--out", default="paper/figs/fig_separability_lda.pdf")
    ap.add_argument("--pca-dim", type=int, default=200,
                     help="PCA dimension before LDA for the high-dim projected "
                          "features (raw features, 768-dim, never need this).")
    args = ap.parse_args()

    data = np.load(args.inp)
    raw_feats = data["raw_feats"].astype(np.float64)
    proj_feats = data["proj_feats"].astype(np.float64)
    task_labels = data["task_labels"]
    n_tasks = len(np.unique(task_labels))

    raw_emb = lda_embed(raw_feats, task_labels, pca_dim=None)
    proj_emb = lda_embed(proj_feats, task_labels, pca_dim=args.pca_dim)

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
        ax.set_xlabel("LDA 1", fontsize=8)
        ax.set_ylabel("LDA 2", fontsize=8)
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

"""Build Figure `separability` (Observation 2) by visualizing the REAL
router's own decision space, not the raw input feature geometry.

Context: three independent honest methods (UMAP, a direct RidgeClassifier
probe, full LDA) all failed to show the published 37.4%/63.9% raw-vs-
projected gap. That's because all three fit a DIRECT 10-way task
classifier on the input vectors, which is a fundamentally weaker design
than what the real router does: a 200-way CLASS-level ridge
(_accumulate_global_ridge), reduced to 10 task scores via per-task
max-pooling (_task_scores_from_logits). That richer decision rule only
exists inside a FITTED classifier's weights -- no visualization of the
raw input geometry can reveal it, regardless of technique.

This script instead replicates that exact mechanism on the extracted
features (which include per-sample class labels, unlike the earlier
version) and visualizes its OUTPUT: the 10-dim task-score vector each
sample gets from the real router, projected to 2D via PCA. This is the
literal space the argmax routing decision is made in, so if the reported
63.9%/37.4% numbers are real, this space should show it -- unlike the
input geometry, which we've now shown three times over does not.

Fits on one half of the extracted samples (mimicking "accumulated training
statistics"), evaluates + visualizes on the held-out half, and prints the
resulting top-1 routing accuracy as a sanity check against the published
37.43% (raw) / 63.86% (projected) before trusting the plot.

Usage:
    python make_separability_figure_router_space.py \
        --in separability_features.npz --out paper/figs/fig_separability_router_space.pdf
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

RIDGE_LAMBDA = 1000.0


def fit_class_ridge(phi_train, y_class_train, n_classes, lam=RIDGE_LAMBDA):
    """Exact replica of models/prism.py::_accumulate_global_ridge +
    _global_ridge_weight: one-hot over ALL classes, ridge closed-form."""
    n = phi_train.shape[0]
    onehot = np.zeros((n, n_classes), dtype=np.float64)
    onehot[np.arange(n), y_class_train] = 1.0
    G = phi_train.T @ phi_train
    C = phi_train.T @ onehot
    G[np.diag_indices_from(G)] += lam
    W = np.linalg.solve(G, C)  # [M, n_classes]
    return W


def task_scores_from_class_logits(logits, classes_per_task):
    """Exact replica of _task_scores_from_logits: max class-logit within
    each task's own contiguous class block."""
    n_tasks = logits.shape[1] // classes_per_task
    scores = np.stack([
        logits[:, t * classes_per_task:(t + 1) * classes_per_task].max(axis=1)
        for t in range(n_tasks)
    ], axis=1)
    return scores


def route_and_score(X, y_class, y_task, n_classes, classes_per_task, seed=0):
    Xtr, Xte, yctr, ycte, yttr, ytte = train_test_split(
        X, y_class, y_task, test_size=0.5, random_state=seed, stratify=y_task,
    )
    W = fit_class_ridge(Xtr.astype(np.float64), yctr, n_classes)
    logits_te = Xte.astype(np.float64) @ W
    task_scores_te = task_scores_from_class_logits(logits_te, classes_per_task)
    pred_task = task_scores_te.argmax(axis=1)
    routing_acc = float((pred_task == ytte).mean()) * 100
    return task_scores_te, ytte, routing_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="separability_features.npz")
    ap.add_argument("--out", default="paper/figs/fig_separability_router_space.pdf")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = np.load(args.inp)
    raw_feats = data["raw_feats"].astype(np.float64)
    proj_feats = data["proj_feats"].astype(np.float64)
    task_labels = data["task_labels"]
    class_labels = data["class_labels"]
    n_classes = int(data["total_classes"][0])
    classes_per_task = int(data["classes_per_task"][0])
    n_tasks = len(np.unique(task_labels))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    cmap = plt.get_cmap("tab10" if n_tasks <= 10 else "tab20")

    for ax, X, title in [
        (axes[0], raw_feats, "(a) Raw [CLS] features"),
        (axes[1], proj_feats, "(b) Projected + ReLU features"),
    ]:
        task_scores, y_true, routing_acc = route_and_score(
            X, class_labels, task_labels, n_classes, classes_per_task, seed=args.seed,
        )
        print(f"{title}: replicated routing accuracy = {routing_acc:.2f}%")
        emb = PCA(n_components=2, random_state=0).fit_transform(task_scores)
        for t in range(n_tasks):
            mask = y_true == t
            ax.scatter(emb[mask, 0], emb[mask, 1], s=6, alpha=0.7,
                       color=cmap(t % cmap.N), label=f"T{t+1}")
        ax.set_title(f"{title}\nreplicated routing acc: {routing_acc:.1f}%", fontsize=10)
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

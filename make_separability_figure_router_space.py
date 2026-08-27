"""Build Figure `separability` (Observation 2) by visualizing the REAL
router's own decision space, not the raw input feature geometry.

Context: three independent honest methods (UMAP, a direct RidgeClassifier
probe, full LDA) all failed to show the published 37.4%/63.9% raw-vs-
projected gap on input features. That's because all three fit a DIRECT
10-way task classifier, a fundamentally weaker design than what the real
router does: a 200-way CLASS-level ridge (_accumulate_global_ridge),
reduced to 10 task scores via per-task max-pooling
(_task_scores_from_logits). A first attempt at replicating that exact
mechanism (fit+evaluate on a random 50/50 split of the same capped
500-samples/task TRAINING pool) got the right *direction* for the first
time (57.5% raw < 60.2% proj) but not the right *magnitude* -- collapsing
most of the real 26.5-point gap, because fitting and evaluating on the
same-distribution training-derived split is an easier, different problem
than the real train-vs-test protocol.

This version fits on the FULL per-task training pool (extract_separability
_features.py's train_* arrays, uncapped) and evaluates + visualizes on the
REAL cumulative test set (test_* arrays) -- the exact split the published
37.43%/63.86% numbers come from. Prints the reproduced routing accuracy
before trusting the plot; if it lands close to those numbers, the
visualization below is finally trustworthy.

Usage:
    python make_separability_figure_router_space.py \
        --in separability_features_full.npz \
        --out paper/figs/fig_separability_router_space.pdf
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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


def route_and_score(X_train, y_class_train, X_test, y_task_test, n_classes, classes_per_task):
    W = fit_class_ridge(X_train.astype(np.float64), y_class_train, n_classes)
    logits_te = X_test.astype(np.float64) @ W
    task_scores_te = task_scores_from_class_logits(logits_te, classes_per_task)
    pred_task = task_scores_te.argmax(axis=1)
    routing_acc = float((pred_task == y_task_test).mean()) * 100
    return task_scores_te, routing_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="separability_features_full.npz")
    ap.add_argument("--out", default="paper/figs/fig_separability_router_space.pdf")
    args = ap.parse_args()

    data = np.load(args.inp)
    n_classes = int(data["total_classes"][0])
    classes_per_task = int(data["classes_per_task"][0])
    test_task = data["test_task_labels"]
    n_tasks = len(np.unique(test_task))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    cmap = plt.get_cmap("tab10" if n_tasks <= 10 else "tab20")

    for ax, feat_key, title in [
        (axes[0], "raw", "(a) Raw [CLS] features"),
        (axes[1], "proj", "(b) Projected + ReLU features"),
    ]:
        X_train = data[f"train_{feat_key}_feats"]
        y_class_train = data["train_class_labels"]
        X_test = data[f"test_{feat_key}_feats"]

        task_scores, routing_acc = route_and_score(
            X_train, y_class_train, X_test, test_task, n_classes, classes_per_task,
        )
        print(f"{title}: replicated routing accuracy = {routing_acc:.2f}%  "
              f"(train n={X_train.shape[0]}, test n={X_test.shape[0]})")
        emb = LinearDiscriminantAnalysis(n_components=2).fit_transform(task_scores, test_task)
        for t in range(n_tasks):
            mask = test_task == t
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

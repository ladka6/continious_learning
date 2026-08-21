"""Emit per-stage last-accuracy curves (cnn_top1) for the Table-1 figure.

For every (model, dataset) it averages cnn_top1 at each incremental stage over
the 5 canonical shuffled seeds (same run selection as aggregate_results.py:
skip the eval_shuffle=false 'bench_' runs and the 'profile' runs, keep the
furthest run per seed), then prints a pgfplots line

    \\addplot+[] coordinates {(x0, acc0) (x1, acc1) ...};  % <model>

with x = the cumulative number of classes seen (init_cls + i*increment),
grouped per dataset in the subplot order of the figure. Paste each dataset's
block into the matching \\nextgroupplot, in the METHOD_ORDER order.

Usage (from continue_learning/, both repos' logs visible):
    python extract_curves.py --roots logs ../pilot_baselines/logs ../tosca-eval/logs
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

PROFILE_PREFIX = "profile"

# dataset meta-name -> (figure title, init_cls, increment)
DATASETS = {
    "cifar224":      ("CIFAR B0 Inc5",        5,  5),
    "cub":           ("CUB B0 Inc10",         10, 10),
    "imagenetr":     ("ImageNet-R B0 Inc20",  20, 20),
    "imageneta":     ("ImageNet-A B0 Inc20",  20, 20),
    "omnibenchmark": ("Omnibenchmark B0 Inc30", 30, 30),
    "vtab":          ("VTAB B0 Inc10",        10, 10),
}
DATASET_ORDER = ["cifar224", "cub", "imagenetr", "imageneta", "omnibenchmark", "vtab"]

# meta model_name -> figure label, in the plotting order of the cycle list
# (last entry = "ours", the bold black curve).
METHOD_ORDER = [
    ("simplecil",     "SimpleCIL"),
    ("ranpac",        "RanPAC"),
    ("l2p",           "L2P"),
    ("dualprompt",    "DualPrompt"),
    ("coda_prompt",   "CODAPrompt"),
    ("aper_adapter",  "APER-Adapter"),
    ("ease",          "EASE"),
    ("mos",           "MOS"),
    ("tosca",         "TOSCA"),
    ("prism",         "PRISM (ours)"),
]
# accept a few alias spellings seen across the PILOT configs
ALIASES = {
    "codaprompt": "coda_prompt",
    "aperadapter": "aper_adapter",
    "aper": "aper_adapter",
    "simple_cil": "simplecil",
}


def norm_model(name):
    n = str(name).lower()
    if n == "tosca":
        # continue_learning's model_name "tosca" == PRISM; the *real* TOSCA
        # lives in the tosca-eval repo. Disambiguate by log root below, not
        # here -- see load().
        return n
    return ALIASES.get(n, n)


def load(roots):
    """-> {(model, dataset): {seed: curve[list of cnn_top1]}}"""
    runs = defaultdict(lambda: defaultdict(dict))  # key -> seed -> (ntasks, curve)
    for root in roots:
        is_tosca_eval = "tosca-eval" in root
        for path in glob.glob(os.path.join(root, "**", "*_metrics.json"), recursive=True):
            base = os.path.basename(path)
            if base.startswith("bench_") or base.startswith(PROFILE_PREFIX):
                continue
            try:
                run = json.load(open(path))
            except (OSError, json.JSONDecodeError):
                print(f"# skipping unreadable {path}")
                continue
            meta, tasks = run.get("meta", {}), run.get("tasks", [])
            if not tasks:
                continue
            model = norm_model(meta.get("model_name"))
            # continue_learning "tosca" == PRISM; tosca-eval "tosca" == TOSCA
            if model in ("tosca", "prism"):
                model = "tosca" if is_tosca_eval else "prism"
            dataset = str(meta.get("dataset"))
            curve = [t.get("cnn_top1") for t in tasks if t.get("cnn_top1") is not None]
            if not curve:
                continue
            seed = meta.get("seed")
            key = (model, dataset)
            prev = runs[key].get(seed)
            if prev is None or len(curve) > len(prev):
                runs[key][seed] = curve
    return runs


def mean_curve(seed_curves):
    """Per-stage mean over seeds; stages present in >=1 seed."""
    if not seed_curves:
        return []
    n = max(len(c) for c in seed_curves.values())
    out = []
    for i in range(n):
        vals = [c[i] for c in seed_curves.values() if i < len(c)]
        out.append(float(np.mean(vals)) if vals else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+",
                    default=["logs", "../pilot_baselines/logs", "../tosca-eval/logs"])
    args = ap.parse_args()
    runs = load(args.roots)

    for ds in DATASET_ORDER:
        title, init_cls, inc = DATASETS[ds]
        print(f"\n% ===== {title}  ({ds}) =====")
        for model, label in METHOD_ORDER:
            seed_curves = runs.get((model, ds), {})
            curve = mean_curve(seed_curves)
            nseeds = len(seed_curves)
            if not curve:
                print(f"        \\addplot+[] coordinates {{}};  % {label} (NO DATA)")
                continue
            coords = " ".join(
                f"({init_cls + i * inc}, {v:.2f})"
                for i, v in enumerate(curve) if v is not None
            )
            print(f"        \\addplot+[] coordinates {{{coords}}};  "
                  f"% {label}  (n={nseeds} seeds, final={curve[-1]:.2f})")


if __name__ == "__main__":
    main()

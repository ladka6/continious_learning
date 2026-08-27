"""Emit the per-stage routing-accuracy curve for Figure `router-forgetting`
(closed-form ridge router vs. a gradient-trained gate on ImageNet-A).

Reads the SAME SLURM logs aggregate_router_ablation.py already uses
(logs/prism-abl-<tag>-<jobid>_<seedidx>.out, written by
submit_router_ablation.sh / run_router_ablation.sh) -- no new run is
needed, since exps/ablation/router_relu15k.json and the router_gate_*.json
configs already set log_routing_acc=true, which makes models/prism.py print
one "Routing accuracy (top-1 task): X" line per incremental stage. That
aggregator only keeps the LAST such line per file (the final-stage number
used in Table `router-ablation`); this script keeps ALL of them, in order,
to reconstruct the full per-stage curve for both router variants, then
averages across the 5 seeds and reports the standard deviation per stage
for the shaded band in the figure.

Usage (run on Snellius, from continious_learning/, where logs/ lives):
    python extract_router_forgetting_curve.py --ridge-tag relu15k --gate-tag gate_linrelu
"""
import argparse
import glob
import re
import numpy as np

ROUTE_ACC_RE = re.compile(r"Routing accuracy \(top-1 task\): ([\d.]+)")


def _latest_array_files(tag):
    """Same de-duplication as aggregate_router_ablation.py: keep only the
    most recent SLURM array (highest job id) for this tag, so a rerun's
    logs are not averaged together with the stale run they replaced."""
    jobs = {}
    for f in glob.glob(f"logs/prism-abl-{tag}-*_*.out"):
        m = re.search(rf"prism-abl-{re.escape(tag)}-(\d+)_\d+\.out$", f)
        if m:
            jobs.setdefault(int(m.group(1)), []).append(f)
    return sorted(jobs[max(jobs)]) if jobs else []


def per_stage_curves(tag):
    """-> list of per-seed curves, each a list of per-stage routing accuracies."""
    curves = []
    for f in _latest_array_files(tag):
        text = open(f).read()
        vals = [float(x) for x in ROUTE_ACC_RE.findall(text)]
        if vals:
            curves.append(vals)
    return curves


def mean_std_per_stage(curves):
    if not curves:
        return [], []
    n_stages = max(len(c) for c in curves)
    means, stds = [], []
    for i in range(n_stages):
        vals = [c[i] for c in curves if i < len(c)]
        means.append(float(np.mean(vals)) if vals else None)
        stds.append(float(np.std(vals)) if vals else None)
    return means, stds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ridge-tag", default="relu15k",
                     help="exps/ablation/router_<tag>.json prefix suffix for the ridge router")
    ap.add_argument("--gate-tag", default="gate_linrelu",
                     help="exps/ablation/router_<tag>.json prefix suffix for the learned gate")
    ap.add_argument("--json-out", default=None,
                     help="Also write {ridge,gate: {mean:[...], std:[...]}} here, "
                          "for make_router_forgetting_figure.py to plot locally.")
    args = ap.parse_args()

    json_payload = {}
    for tag, label in [(args.ridge_tag, "ridge"), (args.gate_tag, "gate")]:
        curves = per_stage_curves(tag)
        n = len(curves)
        means, stds = mean_std_per_stage(curves)
        print(f"\n% ===== {label} (tag={tag}, n_seeds={n}) =====")
        if not means:
            print(f"%   NO DATA -- check logs/prism-abl-{tag}-*_*.out exists")
            continue
        for i, (m, s) in enumerate(zip(means, stds)):
            print(f"%   stage {i+1}: mean={m:.2f}  std={s:.2f}")
        coords_mean = " ".join(f"({i+1}, {m:.2f})" for i, m in enumerate(means))
        coords_upper = " ".join(f"({i+1}, {m+s:.2f})" for i, (m, s) in enumerate(zip(means, stds)))
        coords_lower = " ".join(f"({i+1}, {m-s:.2f})" for i, (m, s) in enumerate(zip(means, stds)))
        print(f"\\addplot+[name path={label}_mean] coordinates {{{coords_mean}}};  % {label} mean")
        print(f"\\addplot+[name path={label}_upper, draw=none] coordinates {{{coords_upper}}};  % {label} +1 std")
        print(f"\\addplot+[name path={label}_lower, draw=none] coordinates {{{coords_lower}}};  % {label} -1 std")
        print(f"\\addplot[fill=gray!20] fill between[of={label}_upper and {label}_lower];  % {label} shaded band")
        json_payload[label] = {"mean": means, "std": stds, "n_seeds": n}

    if args.json_out:
        import json
        with open(args.json_out, "w") as f:
            json.dump(json_payload, f, indent=2)
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()

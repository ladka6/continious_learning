#!/bin/bash
# One go: submit every router variant (each a 3-seed array) with a distinct
# job name so logs/prism-abl-<variant>-* stay separate.
for cfg in exps/ablation/router_*.json; do
  name="prism-abl-$(basename "$cfg" .json | sed 's/^router_//')"
  sbatch --job-name="$name" run_router_ablation.sh "$cfg"
done

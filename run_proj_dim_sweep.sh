#!/bin/bash
#SBATCH --job-name=tosca-proj-dim-sweep
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "$HOME/continious_learning"
source .venv/bin/activate

# Forward-passes-only sweep -- no retraining, so this needs a much shorter
# time budget than a real training job (2h is generous).
#
# Positional args (NOT --export -- its Name=Value,Name=Value syntax splits on
# every comma, silently mangling a comma-separated PROJ_DIMS list):
#   sbatch run_proj_dim_sweep.sh [CONFIG] [PROJ_DIMS] [TASK] [LAMBDA]
# Example:
#   sbatch run_proj_dim_sweep.sh exps/tosca_ina_ridge_router.json 2000,5000,10000 -1 1000
CONFIG="${1:-exps/tosca_ina_ridge_router.json}"
PROJ_DIMS="${2:-2000,5000,10000}"
TASK="${3:--1}"
LAMBDA="${4:-}"
LAMBDA_ARGS=()
if [ -n "$LAMBDA" ]; then
  LAMBDA_ARGS=(--lambda "$LAMBDA")
fi

python sweep_proj_dim.py --config "$CONFIG" --proj-dims "$PROJ_DIMS" --task "$TASK" "${LAMBDA_ARGS[@]}"

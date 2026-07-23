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
# Override via: sbatch --export=CONFIG=...,PROJ_DIMS=...,TASK=...,LAMBDA=... run_proj_dim_sweep.sh
CONFIG="${CONFIG:-exps/tosca_ina_ridge_router.json}"
PROJ_DIMS="${PROJ_DIMS:-2000,5000,10000}"
TASK="${TASK:--1}"
LAMBDA_ARGS=()
if [ -n "${LAMBDA:-}" ]; then
  LAMBDA_ARGS=(--lambda "${LAMBDA}")
fi

python sweep_proj_dim.py --config "$CONFIG" --proj-dims "$PROJ_DIMS" --task "$TASK" "${LAMBDA_ARGS[@]}"

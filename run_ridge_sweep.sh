#!/bin/bash
#SBATCH --job-name=tosca-ridge-sweep
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "$HOME/continious_learning"
source .venv/bin/activate

# Lambda-only re-solve from saved G/C -- no retraining, so 1h is generous.
#
# Positional args (NOT --export -- its Name=Value,Name=Value syntax splits on
# every comma, silently mangling a comma-separated LAMBDAS list):
#   sbatch run_ridge_sweep.sh [CONFIG] [LAMBDAS] [TASK]
# Example:
#   sbatch run_ridge_sweep.sh exps/tosca_ina_ridge_router.json 10,100,1000,10000,100000 -1
CONFIG="${1:-exps/tosca_ina_ridge_router.json}"
LAMBDAS="${2:-10,100,1000,10000,100000}"
TASK="${3:--1}"

python sweep_ridge.py --config "$CONFIG" --lambdas "$LAMBDAS" --task "$TASK"

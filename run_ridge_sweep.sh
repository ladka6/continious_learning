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
# Override via: sbatch --export=CONFIG=...,LAMBDAS=...,TASK=... run_ridge_sweep.sh
CONFIG="${CONFIG:-exps/tosca_ina_ridge_router.json}"
LAMBDAS="${LAMBDAS:-10,100,1000,10000,100000}"
TASK="${TASK:--1}"

python sweep_ridge.py --config "$CONFIG" --lambdas "$LAMBDAS" --task "$TASK"

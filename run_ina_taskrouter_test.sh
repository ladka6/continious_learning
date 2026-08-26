#!/bin/bash
#SBATCH --job-name=prism-ina-taskrouter-test
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "$HOME/continious_learning"
source .venv/bin/activate

# Single-seed smoke test for router_label_space="task" (the literal Eq. 1-2
# router: ridge fit on task-label one-hots, T columns) before committing to
# the full 6-dataset x 5-seed sweep.
python main.py --config exps/prism_ina_taskrouter_test.json --seed 1993

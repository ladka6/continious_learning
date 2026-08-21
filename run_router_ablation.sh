#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
cd "$HOME/continious_learning"
source .venv/bin/activate
SEED=$((1993 + SLURM_ARRAY_TASK_ID))
python main.py --config "$1" --seed "$SEED"

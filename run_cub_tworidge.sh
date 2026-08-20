#!/bin/bash
#SBATCH --job-name=prism-tworidge-cub
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

cd "$HOME/continious_learning"
source .venv/bin/activate

# One seed per array job (1993..1997); a crashed seed never takes the others down.
SEED=$((1993 + SLURM_ARRAY_TASK_ID))
python main.py --config exps/prism_cub_tworidge.json --seed "$SEED"

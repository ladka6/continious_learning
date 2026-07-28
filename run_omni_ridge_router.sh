#!/bin/bash
#SBATCH --job-name=tosca-omni-ridge-router
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

cd "$HOME/continious_learning"
source .venv/bin/activate

# One seed per array job (1993..1997); a crashed seed never takes the others down.
SEED=$((1993 + SLURM_ARRAY_TASK_ID))
python main.py --config exps/tosca_omni_ridge_router.json --seed "$SEED"

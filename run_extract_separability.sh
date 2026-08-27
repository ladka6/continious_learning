#!/bin/bash
#SBATCH --job-name=prism-extract-separability
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "$HOME/continious_learning"
source .venv/bin/activate

# No training: reuses the already-trained abl_relu15k/seed1993 checkpoint's
# AdaptMLP and replays task boundaries offline (see the module docstring).
python extract_separability_features.py \
    --config exps/ablation/router_relu15k.json --seed 1993 \
    --n-per-task 150 --out separability_features.npz

#!/bin/bash
#SBATCH --job-name=prism-flops-imageneta
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "$HOME/continious_learning"
source .venv/bin/activate

# Two profiling runs at DIFFERENT epoch counts (1 and 2): FLOPs are linear
# in the epoch count per task, so aggregate_results.py can solve
# F(e) = fixed + e * per_epoch per task exactly, instead of rescaling the
# whole measured cost (including the epoch-independent replace_fc /
# ridge-accumulation work) by the full epoch ratio.
python main.py --config exps/profile_flops/prism_imageneta_e1.json --seed 1993
python main.py --config exps/profile_flops/prism_imageneta_e2.json --seed 1993

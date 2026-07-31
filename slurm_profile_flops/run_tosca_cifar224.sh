#!/bin/bash
#SBATCH --job-name=tosca-flops-cifar224
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=14:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "$HOME/continious_learning"
source .venv/bin/activate

python main.py --config exps/profile_flops/tosca_cifar224.json --seed 1993

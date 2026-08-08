#!/bin/bash
#SBATCH --job-name=tosca-vtab-headlr-test
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

# Single-seed sanity check (seed 1993 only) for head_lr_multiplier: does
# giving the per-task head a higher learning rate than the tosca adapter
# recover VTAB accuracy? See exps/tosca_vtab_headlr_test.json.
python main.py --config exps/tosca_vtab_headlr_test.json --seed 1993

#!/bin/bash
#SBATCH --job-name=tosca-vtab-headridge-test
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

# Single-seed sanity check (seed 1993 only) for head_use_ridge_features:
# does classifying in the same random-projected, ReLU-expanded space the
# global router uses (instead of raw tosca features) recover VTAB accuracy?
# See exps/tosca_vtab_headridge_test.json.
python main.py --config exps/tosca_vtab_headridge_test.json --seed 1993

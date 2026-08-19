#!/bin/bash
#SBATCH --job-name=prism-ina-shared-head-test
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

# Single-seed sanity check (seed 1993 only) for use_shared_head: one growing
# CosineLinear head (trained jointly with TOSCA via cross-entropy over ALL
# classes seen so far, then this task's rows overwritten with class-mean
# prototypes), like the original TOSCA paper's classifier -- instead of
# Prism's usual independent per-task heads. The ridge router is unchanged.
# See exps/prism_ina_shared_head_test.json and models/prism.py
# (_train_shared_head, _refresh_shared_head_prototypes).
python main.py --config exps/prism_ina_shared_head_test.json --seed 1993

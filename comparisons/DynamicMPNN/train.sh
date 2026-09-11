#!/bin/bash
set -e
mkdir -p logs/

python -m ruff check . --select F821,E9 || exit 1

tar -xzf DynamicMPNN.tar.gz

# api_keys.txt: line 1 = HF_TOKEN, line 2 = WANDB_KEY (see ../../scripts/train_residue_classifier.sh)
export WANDB_KEY=$(tail -n 1 api_keys.txt)

python train.py "$@"

#!/bin/bash
python -m ruff check . --select F821,E9 || exit 1

ls -R
HF_KEY=$(head -n 1 api_keys.txt)
export HF_TOKEN=$HF_KEY
WANDB_KEY=$(tail -n 1 api_keys.txt)
export WANDB_KEY=$WANDB_KEY

python train_residue_classifier.py "$@"
nvidia-smi


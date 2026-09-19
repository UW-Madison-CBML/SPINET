#!/bin/bash
# Entry point run inside the DynamicMPNN Docker container by HTCondor for the k sweep.
#
# Identical to train.sh except that it runs sweep_k.py (which trains one model per k over a
# shared conformer pool) instead of train.py. Everything after 'sweep_k.sh' is forwarded
# straight through, e.g.:
#   ./sweep_k.sh --from-scratch --ds-name atlas --k-grid 2,3,4,5,6,8,10,12,16,20,24,32
set -e
mkdir -p logs/

python -m ruff check . --select F821,E9 || exit 1

tar -xzf DynamicMPNN.tar.gz

# api_keys.txt: line 1 = HF_TOKEN, line 2 = WANDB_KEY (see ../../scripts/train_residue_classifier.sh)
export HF_TOKEN=$(head -n 1 api_keys.txt)
export WANDB_KEY=$(tail -n 1 api_keys.txt)

python sweep_k.py "$@"

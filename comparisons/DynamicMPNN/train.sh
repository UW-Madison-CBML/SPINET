#!/bin/bash
# Entry point run inside the DynamicMPNN Docker container by HTCondor.
#
# Everything after 'train.sh' is forwarded straight to train.py, e.g.:
#   ./train.sh --from-scratch --ds-name mdcath
#
# ATLAS and mdCATH are trained and evaluated separately, one job each (--ds-name),
# matching scripts/train_residue_classifier.py.
set -e
mkdir -p logs/

python -m ruff check . --select F821,E9 || exit 1

tar -xzf DynamicMPNN.tar.gz

# api_keys.txt: line 1 = HF_TOKEN, line 2 = WANDB_KEY (see ../../scripts/train_residue_classifier.sh)
export HF_TOKEN=$(head -n 1 api_keys.txt)
export WANDB_KEY=$(tail -n 1 api_keys.txt)

python train.py "$@"

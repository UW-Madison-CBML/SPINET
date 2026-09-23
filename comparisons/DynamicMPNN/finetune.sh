#!/bin/bash
# Entry point run inside the DynamicMPNN Docker container by HTCondor.
#
# Everything after 'finetune.sh' is forwarded straight to finetune.py, e.g.:
#   ./finetune.sh --ds-name atlas --ckpt DynamicMPNN/checkpoints/single_chain_k5.ckpt
#
# The released checkpoints ship inside DynamicMPNN.tar.gz, so the untar below is what puts
# the starting weights on disk.
set -e
# weights/ is created up front: finetune.sub lists it in transfer_output_files, and a missing
# output holds the job without transferring the logs that say why it failed.
mkdir -p logs/ weights/

python -m ruff check . --select F821,E9 || exit 1

tar -xzf DynamicMPNN.tar.gz

# api_keys.txt: line 1 = HF_TOKEN, line 2 = WANDB_KEY (see ../../scripts/train_residue_classifier.sh)
export HF_TOKEN=$(head -n 1 api_keys.txt)
export WANDB_KEY=$(tail -n 1 api_keys.txt)

python finetune.py "$@"

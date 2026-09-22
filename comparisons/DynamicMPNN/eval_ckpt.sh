#!/bin/bash
# Entry point run inside the DynamicMPNN Docker container by HTCondor.
#
# Everything after 'eval_ckpt.sh' is forwarded straight to eval_ckpt.py, e.g.:
#   ./eval_ckpt.sh --ds-name mdcath --ckpt DynamicMPNN/checkpoints/single_chain_k5.ckpt
#
# Unlike train.sh this trains nothing -- it scores a released checkpoint zero-shot. The
# checkpoints ship inside DynamicMPNN.tar.gz, so the untar below is what puts them on disk.
set -e
mkdir -p logs/

python -m ruff check . --select F821,E9 || exit 1

tar -xzf DynamicMPNN.tar.gz

# api_keys.txt: line 1 = HF_TOKEN, line 2 = WANDB_KEY (see ../../scripts/train_residue_classifier.sh)
export HF_TOKEN=$(head -n 1 api_keys.txt)
export WANDB_KEY=$(tail -n 1 api_keys.txt)

python eval_ckpt.py "$@"

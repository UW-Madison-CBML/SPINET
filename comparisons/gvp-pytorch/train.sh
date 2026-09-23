#!/bin/bash
python -m ruff check . --select F821,E9 || exit 1

WANDB_KEY=$(tail -n 1 api_keys.txt)
export WANDB_KEY=$WANDB_KEY
tar -xvf "$2".tar.gz

tar -xvf gvp-pytorch.tar.gz
mv train.py gvp-pytorch/
cd gvp-pytorch/


python train.py "$@"

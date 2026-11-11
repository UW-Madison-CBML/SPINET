#!/bin/bash
python -m ruff check . --select F821,E9 || exit 1

WANDB_KEY=$(tail -n 1 api_keys.txt)
export WANDB_KEY=$WANDB_KEY

tar -xvf ProteinMPNN.tar.gz
mv train.py ProteinMPNN/
cd ProteinMPNN/

python train.py "$@"

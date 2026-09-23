#!/bin/bash

tar -xvf ScFold.tar.gz
WANDB_KEY=$(tail -n 1 api_keys.txt)
export WANDB_KEY=$WANDB_KEY

cd ScFold/

python3 main.py --epoch 8 "$@"

tar -czvf ../plots.tar.gz plots/

#!/bin/bash

tar -xvf ScFold.tar.gz

#tar -xvf mdcath_pdbs.tar.gz

tar -xvf atlas_pdbs.tar.gz


WANDB_KEY=$(tail -n 1 api_keys.txt)
export WANDB_KEY=$WANDB_KEY

cd ScFold/

python3 main.py --epoch 100 "$@"

tar -czvf ../plots.tar.gz plots/

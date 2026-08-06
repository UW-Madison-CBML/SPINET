#!/bin/bash
ls -R
python -m ruff check . --select F821,E9 || exit 1

tar -I "pigz -p 256" -xvf md_data.tar.gz

rm md_data.tar.gz
HF_KEY=$(head -n 1 api_keys.txt)
export HF_TOKEN=$HF_KEY
WANDB_KEY=$(tail -n 1 api_keys.txt)
export WANDB_KEY=$WANDB_KEY

python train_motion_classifier.py

rm *.ent
rm md_data/*

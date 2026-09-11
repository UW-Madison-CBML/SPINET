#!/bin/bash
mkdir -p logs/

python -m ruff check . --select F821,E9 || exit 1

tar -xvf DynamicMPNN.tar.gz
cd DynamicMPNN/
mv ../train.py ./
mv ../load_dynamics.py ./
mv ../residue_classifier_dataset.py ./
mv ../scrmsd.py ./
tar -xvf ../atlas_index.tar.gz #?


python train.py

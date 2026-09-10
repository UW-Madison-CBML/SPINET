#!/bin/bash
python -m ruff check . --select F821,E9 || exit 1

tar -xvf DynamicMPNN.tar.gz
cd DynamicMPNN/
mv ../train.py ./
tar -xvf ../atlas_index.tar.gz #?


python train.py

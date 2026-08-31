#!/bin/bash
python -m ruff check . --select F821,E9 || exit 1

tar -xvf gvp-pytorch.tar.gz
cd gvp-pytorch/
mv ../train.py ./
tar -xvf ../surffold_data.tar.gz


python train.py

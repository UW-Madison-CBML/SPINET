#!/bin/bash
python -m ruff check . --select F821,E9 || exit 1

tar -xvf gvp-pytorch.tar.gz
mv train.py gvp-pytorch/
cd gvp-pytorch/

python train.py

#!/bin/bash
set -e
mkdir -p logs/

python -m ruff check . --select F821,E9 || exit 1

tar -xzf DynamicMPNN.tar.gz

python train.py

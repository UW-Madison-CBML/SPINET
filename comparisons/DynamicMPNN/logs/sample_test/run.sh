#!/bin/bash
set -e
tar -xzf DynamicMPNN.tar.gz
WANDB_MODE=disabled python test_eval.py 2>&1 | grep -v "it/s\]" 

#!/bin/bash
# run_spinet_on.sh

python -m ruff check . --select F821,E9 || exit 1

ls -R
HF_KEY=$(head -n 1 api_keys.txt)
export HF_TOKEN=$HF_KEY

python run_spinet_on.py "$@"

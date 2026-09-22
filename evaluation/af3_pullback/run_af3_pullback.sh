#!/bin/bash
set -e
set -o pipefail

JSON_FILE="$1"

echo "=== GPU ==="
nvidia-smi

echo "=== INPUT FILES ==="
ls -lh

mkdir -p models af3_output
mv af3.bin.zst models/

python /app/alphafold/run_alphafold.py \
    --json_path="${JSON_FILE}" \
    --model_dir=models \
    --output_dir=af3_output \
    --run_data_pipeline=false

echo "=== OUTPUT ==="
find af3_output -maxdepth 3 -type f | sort
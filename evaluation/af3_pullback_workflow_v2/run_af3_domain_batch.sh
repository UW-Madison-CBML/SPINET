#!/bin/bash
set -euo pipefail

PACKAGE_ID="$1"
PACKAGE_TAR="$2"

echo "=== ${PACKAGE_ID} ==="
echo "Host: $(hostname)"
date
nvidia-smi

if [[ ! -f "${PACKAGE_TAR}" ]]; then
    PACKAGE_TAR="$(basename "${PACKAGE_TAR}")"
fi

if [[ ! -f "${PACKAGE_TAR}" ]]; then
    echo "ERROR: package tar not found: ${PACKAGE_TAR}" >&2
    ls -lah >&2
    exit 2
fi

if [[ ! -f af3.bin.zst ]]; then
    echo "ERROR: af3.bin.zst not found in scratch" >&2
    ls -lah >&2
    exit 2
fi

mkdir -p work models af3_output
tar -xzf "${PACKAGE_TAR}" -C work
ln -sf "${PWD}/af3.bin.zst" models/af3.bin.zst

echo "Input JSON count: $(find work/inputs -maxdepth 1 -name '*.json' | wc -l)"
echo "Template CIF count: $(find work/inputs -maxdepth 1 -name '*.cif' | wc -l)"

python /app/alphafold/run_alphafold.py   --input_dir=work/inputs   --model_dir=models   --output_dir=af3_output   --run_data_pipeline=false

tar -czf "af3_results_${PACKAGE_ID}.tar.gz" af3_output

echo "=== Completed ${PACKAGE_ID} ==="
du -sh "af3_results_${PACKAGE_ID}.tar.gz"
date

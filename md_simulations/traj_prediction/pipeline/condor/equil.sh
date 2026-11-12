#!/bin/bash
# Build the solvated system from a folded model, then equilibrate it.
# Usage: equil.sh <pred|native> <equil_ns> <seed>
set -euo pipefail
export HOME=$PWD
WHO=$1; EQUIL_NS=$2; SEED=$3
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || true
python3 build_system.py --pdb "fold/${WHO}_af2.pdb" --out-dir "equil_${WHO}"
python3 run_md.py equil --build-dir "equil_${WHO}" --equil-ns "$EQUIL_NS" --seed "$SEED"

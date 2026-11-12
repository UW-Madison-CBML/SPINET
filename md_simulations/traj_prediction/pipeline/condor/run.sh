#!/bin/bash
# Generic job wrapper: runs the command given as arguments in the job's scratch folder.
set -euo pipefail
export HOME=$PWD                 # HTMD and matplotlib need a writable home
export MPLCONFIGDIR=$PWD/.mpl
mkdir -p "$MPLCONFIGDIR"
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || true
exec "$@"

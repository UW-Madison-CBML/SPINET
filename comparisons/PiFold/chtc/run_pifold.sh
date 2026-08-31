#!/bin/bash
# Entry point run inside the PiFold Docker container by HTCondor.
#
# HTCondor's file transfer drops the code (API/, methods/, utils/,
# main.py, parser.py) and, if used, a data tarball into the job's
# scratch dir (the container's working dir), then runs this script.
# Everything after 'run_pifold.sh' in the submit file's `arguments`
# line is forwarded straight to main.py, e.g.:
#
#   arguments = --data_name CATH --data_root ./data/ --batch_size 8 --epoch 100
#
set -euo pipefail

echo ">>> host:   $(hostname)"
echo ">>> gpu:    $(nvidia-smi -L 2>/dev/null || echo 'no GPU visible')"
python -c "import torch; print('>>> torch:', torch.__version__, 'cuda available:', torch.cuda.is_available())"

if [[ -f surffold_data.tar.gz && ! -d surffold_data ]]; then
    echo ">>> extracting surffold_data.tar.gz"
    tar -xzf surffold_data.tar.gz
fi

mkdir -p results

echo ">>> running: python main.py $*"
exec python main.py "$@"

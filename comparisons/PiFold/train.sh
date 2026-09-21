#!/bin/bash
# Entry point run inside the PiFold Docker container by HTCondor (or locally).
#
# PiFold.tar.gz should be a tarball of a *pristine* checkout of A4Bio/PiFold
# (`git clone https://github.com/A4Bio/PiFold && tar czf PiFold.tar.gz PiFold/`).
# This script extracts it and copies this folder's dataset/wandb overlay
# (API/, main.py, parser.py) plus the shared lib/ modules on top of it, so the
# upstream repo itself is never modified -- same pattern as ../MapDiff.
#
# HTCondor's file transfer drops everything flat into the job's scratch dir;
# the split index csv and trajectory store land there too. Everything after
# 'run_pifold.sh' is forwarded straight to main.py, e.g.:
#
#   arguments = --data_name ATLAS --data_root ./ --batch_size 8 --epoch 100
#
set -euo pipefail

mkdir -p logs/

echo ">>> host:   $(hostname)"
echo ">>> gpu:    $(nvidia-smi -L 2>/dev/null || echo 'no GPU visible')"
python -c "import torch; print('>>> torch:', torch.__version__, 'cuda available:', torch.cuda.is_available())"

# api_keys.txt: line 1 = HF_TOKEN, last line = WANDB_KEY (see
# ../../scripts/train_residue_classifier.sh). main.py's --wandb_key_file wants a file
# holding *only* the W&B key, so split it out here.
if [[ -f api_keys.txt ]]; then
    export HF_TOKEN=$(head -n 1 api_keys.txt)
    tail -n 1 api_keys.txt > wandb_api.txt
fi

if [[ ! -d PiFold ]]; then
    echo ">>> extracting PiFold.tar.gz"
    tar -xzf PiFold.tar.gz
fi

# Upstream featurisers call `.astype(np.int)`, an alias numpy removed in 1.24
# ("module 'numpy' has no attribute 'int'", raised inside the DataLoader
# workers on the first batch). `int` is what the alias always meant. Patched
# here rather than shipped as overlay copies of the two files: they are large,
# otherwise untouched, and would silently drift from upstream.
sed -i 's/\.astype(np\.int)/.astype(int)/' PiFold/API/featurizer.py PiFold/API/dataloader_gtrans.py
if grep -rn 'np\.int)' PiFold/API/featurizer.py PiFold/API/dataloader_gtrans.py; then
    echo ">>> np.int patch did not apply -- upstream changed these lines" >&2
    exit 1
fi

echo ">>> overlaying dataset support + wandb wiring onto pristine PiFold/"
cp -r API main.py parser.py PiFold/
for f in wandb_api.txt relaxed_pdb.py dataset_splits.py stats_utils.py \
         atlas_data.h5 atlas_cross_val_index.csv mdcath_spinet_320_0.h5 mdcath_320_0_topology_split.csv; do
    [[ -f "$f" ]] && cp "$f" PiFold/
done

cd PiFold
mkdir -p results

echo ">>> running: python main.py $*"
exec python main.py "$@"

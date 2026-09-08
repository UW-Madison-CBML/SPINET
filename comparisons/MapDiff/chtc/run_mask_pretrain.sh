#!/bin/bash
# Entry point run inside the MapDiff Docker container by HTCondor (or locally),
# for stage 1 of MapDiff training: mask-prior IPA pretraining.
#
# Extracts a *pristine* checkout of peizhenbai/MapDiff (MapDiff.tar.gz) and
# copies ../overlay/ (ATLAS dataset support + wandb wiring) on top of it --
# MapDiff/ itself is never modified, so re-cloning/re-pulling it always works.
# See ../README.md.
#
# Everything after 'run_mask_pretrain.sh' in the submit file's `arguments`
# line is forwarded straight to mask_ipa_pretrain.py, e.g.:
#
#   arguments = --config-name=mask_pretrain dataset=atlas ...
#
set -euo pipefail

echo ">>> host:   $(hostname)"
echo ">>> gpu:    $(nvidia-smi -L 2>/dev/null || echo 'no GPU visible')"
python -c "import torch; print('>>> torch:', torch.__version__, 'cuda available:', torch.cuda.is_available())"

if [[ ! -d MapDiff ]]; then
    echo ">>> extracting MapDiff.tar.gz"
    tar -xzf MapDiff.tar.gz
fi

echo ">>> overlaying ATLAS dataset support + wandb wiring onto pristine MapDiff/"
cp -r overlay/. MapDiff/
for f in wandb_api.txt stats_utils.py; do
    [[ -f "$f" ]] && cp "$f" MapDiff/
done

if [[ -f surffold_data.tar.gz && ! -d MapDiff/surffold_data ]]; then
    echo ">>> extracting surffold_data.tar.gz"
    tar -xzf surffold_data.tar.gz -C MapDiff/
fi

cd MapDiff
if [[ ! -d surffold_data/atlas_process/train ]]; then
    echo ">>> featurizing raw ATLAS train structures into MapDiff graphs (runs DSSP once per protein)"
    # Stage 1 only trains on the train split, so skip featurizing validation/test here.
    python data/generate_graph_atlas.py --pdb_root surffold_data --save_root surffold_data/atlas_process \
        --splits train --marginal_out surffold_data/train_marginal_x_atlas.pt
fi

echo ">>> running: python mask_ipa_pretrain.py $*"
exec python mask_ipa_pretrain.py "$@"

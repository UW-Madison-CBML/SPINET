#!/bin/bash
# Entry point run inside the MapDiff Docker container by HTCondor (or locally),
# for stage 2 of MapDiff training: the denoising diffusion network, seeded by
# a stage-1 mask-prior IPA checkpoint (see run_mask_pretrain.sh).
#
# Extracts a *pristine* checkout of peizhenbai/MapDiff (MapDiff.tar.gz) and
# copies ../overlay/ (ATLAS dataset support + wandb wiring) on top of it --
# MapDiff/ itself is never modified. See ../README.md.
#
# Everything after 'run_train.sh' in the submit file's `arguments` line is
# forwarded straight to main.py, e.g.:
#
#   arguments = --config-name=diff_config dataset=atlas prior_model.path=./ipa_checkpoint.pt ...
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
# stage-1 IPA checkpoint(s) transferred in flat -- copy alongside so
# `prior_model.path=./<filename>.pt` (set in train.sub's `arguments`)
# resolves once we cd into MapDiff/ below.
cp -f *.pt MapDiff/ 2>/dev/null || true

if [[ -f surffold_data.tar.gz && ! -d MapDiff/surffold_data ]]; then
    echo ">>> extracting surffold_data.tar.gz"
    tar -xzf surffold_data.tar.gz -C MapDiff/
fi

cd MapDiff
if [[ ! -d surffold_data/atlas_process/train ]]; then
    echo ">>> featurizing raw ATLAS structures into MapDiff graphs (runs DSSP once per protein)"
    python data/generate_graph_atlas.py --pdb_root surffold_data --save_root surffold_data/atlas_process \
        --marginal_out surffold_data/train_marginal_x_atlas.pt
fi

echo ">>> running: python main.py $*"
exec python main.py "$@"

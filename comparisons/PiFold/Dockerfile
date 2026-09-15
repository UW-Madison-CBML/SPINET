# PiFold runtime environment for CHTC (Docker universe).
#
# This image only provides the software environment. The PiFold source code and
# datasets are NOT baked in here -- CHTC jobs pull them in at runtime via
# HTCondor file transfer / the /staging area, so the image stays small and
# doesn't need to be rebuilt every time the code changes.
#
# Build:  docker build -t <dockerhub_user>/pifold:latest -f Dockerfile .
# Push:   docker push <dockerhub_user>/pifold:latest
# (see build_and_push.sh for a one-shot script)

# torch is at 2.6 rather than the 2.1.2 this image used to carry because
# transformers' `check_torch_load_is_safe()` refuses `torch.load` on anything
# below that (CVE-2025-32434), and facebook/esmfold_v1 -- which lib/scrmsd.py's
# `load_esmfold` pulls for the self-consistency RMSD metric -- only publishes
# pytorch_model.bin, never a safetensors copy. Same bump as ../MapDiff; cu118
# is kept so the CUDA runtime the CHTC nodes see does not change.
FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Scope of the dependency list below is everything run_pifold.sh touches:
#
#  * the pristine upstream PiFold/ checkout (PiFold.tar.gz) -- torch,
#    torch_scatter, numpy, tqdm, and nothing else;
#  * this folder's overlay (main.py, parser.py, API/relaxed_dataset.py);
#  * the shared ../../lib modules the overlay imports -- scrmsd.py,
#    relaxed_pdb.py, dataset_splits.py, stats_utils.py.
#
# Mapped back to the imports that need them:
#   biopython   -- relaxed_pdb.py (Bio.Align/PDB/Data.IUPACData) and
#                  stats_utils.py (Bio.SeqUtils.seq1)
#   requests    -- relaxed_pdb.py, downloading deposited structures from RCSB
#   h5py        -- relaxed_pdb.reference_seqs_from_h5 (trajectory store)
#   pandas      -- dataset_splits.py, reading the split index csv
#   transformers/accelerate/scipy -- lib/scrmsd.py's ESMFold scRMSD metric
#   matplotlib + scikit-learn     -- stats_utils.to_log_dict's confusion matrix
#   wandb       -- main.py's --wandb_key_file run logging
#
# Deliberately NOT here: mdtraj. relaxed_pdb.py re-exports BACKBONE_ATOMS
# precisely so this image never has to import scripts/load_dynamics.py.

# torch_scatter ships compiled kernels and must come from the wheel index built
# for the exact torch+CUDA combo in the FROM line above -- see
# https://data.pyg.org/whl/. Stating the combo once as env vars (the
# ../DynamicMPNN pattern) keeps a base-image bump from silently leaving the
# wheel index pointing at the old ABI.
#
# numpy stays at 1.26.4: upstream PiFold is numpy-1 era code (train.sh already
# has to patch out its `np.int` use, removed in numpy 1.24), and numpy 2 drops
# considerably more of that surface. torch 2.6 itself is happy either way --
# its wheels are built against numpy-2 headers, which stay runtime-compatible
# with numpy 1.x.
ENV TORCH_VERSION=2.6.0
ENV CUDA_TAG=cu118

RUN pip install --no-cache-dir \
        torch_scatter==2.1.2 -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html \
    && pip install --no-cache-dir \
        numpy==1.26.4 \
        biopython==1.83 \
        tqdm==4.66.2 \
        requests \
        h5py \
        pandas \
        scipy \
        scikit-learn \
        matplotlib \
        "transformers==4.55.4" \
        accelerate==1.10.1

# wandb is installed on its own, AFTER the block above, for the same reason as
# in ../DynamicMPNN: listed alongside everything else, pip is free to backtrack
# it down to a release whose `apikey.write_key` enforces `len(key) == 40`, and
# W&B now issues longer keys ("API key must be 40 characters long, yours was
# 86" killed a job that way). 0.22.3 is the first line to relax that check.
RUN pip install --no-cache-dir wandb==0.22.3

# Fail the build (rather than a queued CHTC job that has already claimed a GPU)
# if the resolved wheels don't actually hang together. The failure modes this
# catches, all of which have bitten the comparison images before:
#
#  * torch slipping back below 2.6. `check_torch_load_is_safe()` is called
#    inside EsmForProteinFolding.from_pretrained, and on an older torch it
#    raises rather than loading esmfold_v1's pytorch_model.bin -- calling it
#    here is the same guard the run-time path hits.
#  * transformers silently disabling its torch backend. A transformers whose
#    torch requirement isn't met degrades to a tokenizers-only install, and
#    EsmForProteinFolding then raises "requires the PyTorch library but it was
#    not found" deep inside scrmsd.load_esmfold().
#  * numpy 2.x getting pulled in, which upstream PiFold's numpy-1 era code
#    (see the np.int patch in train.sh) is not written against.
#  * scipy missing -- ESMFold silently falls back to an approximate init.
#  * accelerate missing -- scrmsd.load_esmfold passes low_cpu_mem_usage=True,
#    which transformers 4.x refuses without it.
RUN python -c "\
import numpy, scipy, sklearn, pandas, h5py, requests, tqdm, Bio, torch, transformers, accelerate, wandb; \
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot; \
from transformers.utils import is_torch_available, is_scipy_available, check_torch_load_is_safe; \
from transformers import AutoTokenizer, EsmForProteinFolding; \
from sklearn.metrics import ConfusionMatrixDisplay; \
from Bio import Align, PDB; from Bio.SeqUtils import seq1; \
assert numpy.__version__.startswith('1.'), 'numpy must be 1.x for upstream PiFold, got ' + numpy.__version__; \
assert is_torch_available(), 'transformers ' + transformers.__version__ + ' disabled its torch backend against torch ' + torch.__version__; \
assert is_scipy_available(), 'scipy missing: ESMFold falls back to an approximate init'; \
check_torch_load_is_safe(); \
EsmForProteinFolding.from_pretrained; \
print('>>> ok: torch', torch.__version__, '| transformers', transformers.__version__, '| numpy', numpy.__version__, '| wandb', wandb.__version__)"

# torch_scatter ships compiled kernels that a bare import does not touch, and
# upstream PiFold's prodesign_module.py calls scatter_sum/softmax/mean on every
# forward pass -- run one on real (CPU) tensors so a torch/torch_scatter ABI
# mismatch shows up here instead of mid-epoch.
RUN python -c "\
import torch; from torch_scatter import scatter_sum, scatter_softmax, scatter_mean; \
idx = torch.tensor([0, 0, 1, 1]); \
assert torch.allclose(scatter_sum(torch.ones(4), idx), torch.tensor([2., 2.])); \
scatter_softmax(torch.randn(4), idx); scatter_mean(torch.randn(4), idx); \
print('SCATTER KERNEL OK')"

# The key-length rule lives inside wandb's own source, not in its metadata, so
# the version pin alone does not prove the fix is present. Read the installed
# `write_key` back and require the relaxed wording.
RUN python -c "\
import wandb; from wandb.sdk.lib import apikey; \
src = __import__('inspect').getsource(apikey.write_key); \
assert 'at least 40' in src, 'wandb reintroduced the exact-40-character API key check'; \
print('WANDB LONG KEY OK')"

WORKDIR /workspace

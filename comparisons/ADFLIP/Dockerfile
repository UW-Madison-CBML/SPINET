# ADFLIP runtime environment for CHTC (Docker universe).
#
# This image only provides the software environment. The ADFLIP source, its released checkpoint
# and PIPPack's side-chain weights are NOT baked in here -- eval_pretrained.sh pulls them in at
# run time (ADFLIP.tar.gz plus this folder's eval_pretrained.py / multi_state.py and the shared
# ../../lib modules, transferred in flat by HTCondor / the /staging area), so the image stays
# small and does not need rebuilding when the code changes.
#
# Build:  docker build -t <dockerhub_user>/adflip:latest -f Dockerfile .
# Push:   docker push <dockerhub_user>/adflip:latest
# (see build_and_push.sh for a one-shot script)

# torch stays at upstream's pinned 2.1.0 -- it is NOT bumped to the 2.6 the ../PiFold and
# ../MapDiff images carry, and that difference is load-bearing:
#
#  * torch 2.6 flipped `torch.load`'s `weights_only` default to True. ADFLIP's own
#    `test/design.py` (`torch.load(ckpt_path, map_location="cpu")`) and
#    `DiscreteFlow_AA.load_sc_model` (PIPPack's `*_ckpt.pt` / `*_config.pickle`) both call it
#    bare, and both checkpoints carry pickled *config objects*, not just tensors. On torch >= 2.6
#    those loads raise UnpicklingError before a single residue is sampled.
#  * The reason the other two images needed 2.6 does not apply here: that was
#    transformers' `check_torch_load_is_safe()` refusing to load esmfold_v1, for the scRMSD
#    metric. eval_pretrained.py reports no scRMSD, so there is no ESMFold and no transformers.
#
# cu121 matches upstream's README, which installs torch-cluster/torch-scatter from the
# torch-2.1.0+cu121 wheel index.
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Scope of the dependency list below is everything eval_pretrained.sh touches:
#
#  * the pristine upstream ADFLIP/ checkout -- its requirements.txt, minus the training-only
#    entries, plus torch_scatter/torch_cluster from its README;
#  * the bundled PIPPack/, whose `ensembled_inference`/`inference` modules the adaptive sampler
#    imports for side-chain repacking -- these drag in hydra, omegaconf and lightning;
#  * this folder's multi_state.py + eval_pretrained.py;
#  * the shared ../../lib modules they import.
#
# Mapped back to the imports that need them:
#   prody        -- data/all_atom_parse.py; also *defines the token vocabulary* via
#                   prody.flagDefinition('stdaa') (see the check further down)
#   biopython    -- all_atom_parse.py (Bio.PDB/Data.IUPACData), lib/relaxed_pdb.py,
#                   lib/stats_utils.py (Bio.SeqUtils.seq1)
#   ema-pytorch  -- test/design.py wraps the flow model in EMA to load the released checkpoint
#   hydra-core/omegaconf/lightning/torchmetrics -- PIPPack.ensembled_inference, PIPPack.model.loss
#   torch_geometric + torch_scatter/torch_cluster -- the zoidberg GNN and PIPPack's MPNN
#   scipy        -- all_atom_parse.py (scipy.spatial.cKDTree, the interacting-residue masks)
#   pyyaml       -- model/discrete_flow_aa.py
#   sortedcontainers -- prody
#   h5py         -- lib/traj_frames.py, reading the trajectory store
#   pandas       -- lib/dataset_splits.py, reading the split index csv
#   requests     -- lib/relaxed_pdb.py, downloading deposited structures (--structure-source relaxed)
#   matplotlib + scikit-learn -- lib/stats_utils.to_log_dict's confusion matrix
#   wandb        -- eval_pretrained.py's --wandb-key-file run logging
#
# Deliberately NOT here:
#   transformers/accelerate -- no ESMFold, see the FROM comment
#   mmseqs2     -- upstream needs it only to *build* training clusters; eval_pretrained.py reads
#                  the precomputed data/cluster/*.pkl for its training-overlap report
#   biopandas   -- imported only by PIPPack/data/extract_targets_from_full_targets.py, which is
#                  not on the inference path (PIPPack's subpackages are namespace packages, so
#                  importing PIPPack.data.protein runs no __init__ that would pull it in)

# torch_scatter/torch_cluster ship compiled kernels and must come from the wheel index built for
# the exact torch+CUDA combo in the FROM line above. Stating the combo once as env vars (the
# ../DynamicMPNN pattern) keeps a base-image bump from silently leaving the index on the old ABI.
ENV TORCH_VERSION=2.1.0
ENV CUDA_TAG=cu121

# numpy is held below 2 for prody 2.6.1, which is compiled against the numpy-1 ABI. It is at
# 1.26.4 rather than upstream's pinned 1.24.0: 1.24 predates the pandas 2.2 / scipy 1.14 wheels
# resolved below, and nothing in ADFLIP depends on the difference.
#
# lightning is pinned rather than left open because it requires torch>=2.1 and pip would happily
# satisfy a newer lightning by *upgrading torch* out from under the FROM line -- which would
# reintroduce the weights_only breakage above. The assertion block after this verifies it did not.
RUN pip install --no-cache-dir \
        numpy==1.26.4 \
        torch_scatter==2.1.2 torch_cluster==1.6.3 \
        -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html \
    && pip install --no-cache-dir \
        torch-geometric==2.6.1 \
        prody==2.6.1 \
        biopython==1.84 \
        ema-pytorch==0.4.8 \
        hydra-core==1.3.2 \
        omegaconf==2.3.0 \
        lightning==2.1.3 \
        torchmetrics==1.3.0 \
        scipy==1.11.4 \
        pyyaml==6.0.2 \
        sortedcontainers==2.4.0 \
        h5py \
        pandas \
        requests \
        scikit-learn \
        matplotlib \
        tqdm==4.66.5

# wandb is installed on its own, AFTER the block above, for the same reason as in ../PiFold and
# ../DynamicMPNN: listed alongside everything else, pip is free to backtrack it down to a release
# whose `apikey.write_key` enforces `len(key) == 40`, and W&B now issues longer keys. 0.22.3 is
# the first line to relax that check.
RUN pip install --no-cache-dir wandb==0.22.3

# Fail the build (rather than a queued CHTC job that has already claimed a GPU) if the resolved
# wheels do not actually hang together.
#
#  * torch being upgraded past 2.6 by lightning's resolver, which would make every `torch.load`
#    on the inference path raise -- the failure this image's whole version choice exists to
#    avoid. Checked by calling `torch.load` on a pickled *non-tensor* object exactly as
#    test/design.py does, so the guard matches the run-time path instead of trusting a version
#    string alone.
#  * numpy 2.x getting pulled in, which prody 2.6.1 (numpy-1 ABI) cannot import against.
#  * lightning degrading to a stub, which PIPPack.ensembled_inference imports at module level.
RUN python -c "\
import io, argparse, numpy, scipy, sklearn, pandas, h5py, requests, tqdm, yaml, sortedcontainers; \
import torch, prody, Bio, ema_pytorch, hydra, omegaconf, lightning, torchmetrics, wandb; \
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot; \
from sklearn.metrics import ConfusionMatrixDisplay; \
from Bio import PDB; from Bio.SeqUtils import seq1; from Bio.Data import IUPACData; \
from scipy.spatial import cKDTree; \
assert numpy.__version__.startswith('1.'), 'numpy must be 1.x for prody 2.6.1, got ' + numpy.__version__; \
ckpt = {'config': argparse.Namespace(zoidberg_denoiser=argparse.Namespace(hidden_dim=128)), 'model': {'w': torch.zeros(2)}}; \
buf = io.BytesIO(); torch.save(ckpt, buf); buf.seek(0); \
loaded = torch.load(buf, map_location='cpu'); \
assert loaded['config'].zoidberg_denoiser.hidden_dim == 128, 'bare torch.load lost the pickled config object'; \
print('>>> ok: torch', torch.__version__, '| prody', prody.__version__, '| numpy', numpy.__version__, '| lightning', lightning.__version__)"

# The residue token vocabulary -- and therefore the meaning of every output class of the released
# checkpoint -- is built at import time from `prody.flagDefinition('stdaa')`
# (data/all_atom_parse.py: `add_flag_tokens('stdaa')` after <PAD> and <MASK>). If a prody release
# changed that set or its order, ADFLIP would still load and still produce plausible-looking
# recovery numbers, silently scored against shifted labels. eval_pretrained.py maps tokens to the
# 20-letter alphabet *by name* for this reason and asserts contiguity; this is the same guard at
# build time, on the one input it cannot check without the source tree.
RUN python -c "\
import prody; \
expected = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']; \
stdaa = list(prody.flagDefinition('stdaa')); \
assert sorted(stdaa) == sorted(expected), 'prody stdaa is not the 20 standard residues: ' + repr(stdaa); \
assert len(stdaa) == 20, 'prody stdaa has ' + str(len(stdaa)) + ' entries'; \
nonstd = list(prody.flagDefinition('nonstdaa')); \
assert 'MSE' in nonstd, 'prody nonstdaa no longer carries MSE'; \
print('>>> prody vocabulary ok:', stdaa)"

# torch_scatter/torch_cluster ship compiled kernels that a bare import does not touch, and both
# the zoidberg GNN and PIPPack's MPNN call them on every forward pass -- run them on real (CPU)
# tensors so a torch/torch_scatter ABI mismatch shows up here instead of mid-run.
RUN python -c "\
import torch; \
from torch_scatter import scatter_sum, scatter_mean, scatter_softmax; \
from torch_cluster import knn_graph; \
idx = torch.tensor([0, 0, 1, 1]); \
assert torch.allclose(scatter_sum(torch.ones(4), idx), torch.tensor([2., 2.])); \
scatter_mean(torch.randn(4), idx); scatter_softmax(torch.randn(4), idx); \
e = knn_graph(torch.randn(64, 3), k=8); \
assert e.shape == (2, 64 * 8), e.shape; \
print('>>> kernels ok: torch', torch.__version__, torch.version.cuda)"

# The key-length rule lives inside wandb's own source, not in its metadata, so the version pin
# alone does not prove the fix is present. Read the installed `write_key` back.
RUN python -c "\
import wandb; from wandb.sdk.lib import apikey; \
src = __import__('inspect').getsource(apikey.write_key); \
assert 'at least 40' in src, 'wandb reintroduced the exact-40-character API key check'; \
print('WANDB LONG KEY OK')"

WORKDIR /workspace

# MapDiff runtime environment for CHTC (Docker universe).
#
# This image only provides the software environment (CUDA + PyTorch +
# PyG/torch_scatter/torch_cluster + DSSP + the pure-Python deps MapDiff
# needs, plus transformers for the ESMFold self-consistency RMSD metric).
# The MapDiff source code and datasets are NOT baked in here -- train.sh
# pulls them in at run time (a pristine MapDiff.tar.gz + this folder's
# ATLAS/wandb/scRMSD overlay, transferred in flat by HTCondor / the
# /staging area), so the image stays small and doesn't need to be
# rebuilt every time the code changes.
#
# Build:  docker build -t <dockerhub_user>/mapdiff:latest -f Dockerfile .
# Push:   docker push <dockerhub_user>/mapdiff:latest
# (see build_and_push.sh for a one-shot script)

# torch is at 2.6 rather than the 2.1.2 this image used to carry because
# transformers' `check_torch_load_is_safe()` refuses `torch.load` on anything
# below that (CVE-2025-32434), and facebook/esmfold_v1 -- which lib/scrmsd.py's
# `load_esmfold` pulls for the self-consistency RMSD metric -- only publishes
# pytorch_model.bin, never a safetensors copy. So the scRMSD path cannot load
# the model at all on an older torch; it dies in train.py right after the data
# is featurised.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# mkdssp (DSSP) computes secondary structure in data/generate_graph_relaxed.py
# and data/generate_graph_cath.py (Bio.PDB.DSSP is invoked as dssp='mkdssp').
# The base image ships conda, which is the only reliable way to get a
# prebuilt mkdssp binary.
RUN conda install -y -c ostrokach dssp && conda clean -afy

# torch_scatter/torch_cluster ship compiled kernels and must come from the wheel
# index built for the exact torch+CUDA combo in the FROM line above;
# torch-geometric itself is pure Python. Stating the combo once as env vars (the
# ../DynamicMPNN pattern) keeps a base-image bump from silently leaving the
# wheel index pointing at the old ABI.
#
# pyg now comes from pip instead of `conda install pyg=2.4.0 -c pyg`: the pyg
# conda channel has no builds against torch 2.6, and 2.4.0 predates torch 2.6
# support anyway. 2.6.1 is what ../DynamicMPNN already runs.
ENV TORCH_VERSION=2.6.0
ENV CUDA_TAG=cu124

# numpy is held at 1.x for rdkit 2023.3.3, which is compiled against the numpy-1
# ABI -- *not* for torch any more. torch 2.6 wheels are built against numpy-2
# headers, which stay runtime-compatible with numpy 1.x. It is installed in the
# same step as pyg so torch-geometric's own numpy requirement can never pull a
# 2.x wheel in first.
RUN pip install --no-cache-dir \
        "numpy<2" \
        torch_scatter==2.1.2 torch_cluster==1.6.3 torch-geometric==2.6.1 \
        -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html \
    && pip install --no-cache-dir \
        rdkit==2023.3.3 \
        hydra-core==1.3.2 \
        biopython==1.81 \
        einops==0.7.0 \
        prettytable \
        wandb==0.22.3 \
        "transformers==4.55.4" \
        accelerate==1.10.1 \
        scipy \
        scikit-learn \
        h5py \
        matplotlib \
        pandas \
        tqdm \
        mdtraj==1.10.3 \
        requests \
        ruff

# Fail the build (rather than a 30-minute CHTC job) if the resolved wheels
# don't actually hang together. The failure modes this catches, all of which
# have bitten this image before:
#
#  * torch slipping back below 2.6. `check_torch_load_is_safe()` is called
#    inside EsmForProteinFolding.from_pretrained, and on an older torch it
#    raises rather than loading esmfold_v1's pytorch_model.bin -- calling it
#    here is the same guard the run-time path hits.
#  * transformers silently disabling its torch backend. A transformers whose
#    torch requirement isn't met degrades to a tokenizers-only install, and
#    EsmForProteinFolding then raises "requires the PyTorch library but it was
#    not found" deep inside scrmsd.load_esmfold().
#  * numpy 2.x getting pulled in, which rdkit 2023.3.3 (numpy-1 ABI) cannot
#    import against.
RUN python -c "\
import numpy, scipy, sklearn, torch, transformers, rdkit; \
from transformers.utils import is_torch_available, is_scipy_available, check_torch_load_is_safe; \
from transformers import EsmForProteinFolding; \
assert numpy.__version__.startswith('1.'), 'numpy must be 1.x for rdkit 2023.3.3, got ' + numpy.__version__; \
assert is_torch_available(), 'transformers ' + transformers.__version__ + ' disabled its torch backend against torch ' + torch.__version__; \
assert is_scipy_available(), 'scipy missing: ESMFold falls back to an approximate init'; \
check_torch_load_is_safe(); \
EsmForProteinFolding.from_pretrained; \
print('>>> ok: torch', torch.__version__, '| transformers', transformers.__version__, '| numpy', numpy.__version__, '| sklearn', sklearn.__version__)"

# torch_scatter/torch_cluster ship compiled kernels that a bare import does not
# touch, and MapDiff's EGNN calls scatter on every forward pass -- run one on
# real (CPU) tensors so a torch/torch_scatter ABI mismatch shows up here instead
# of mid-epoch.
RUN python -c "\
import torch, torch_geometric; \
from torch_scatter import scatter_mean; \
from torch_geometric.nn.pool import knn_graph; \
idx = torch.tensor([0, 0, 1, 1]); \
scatter_mean(torch.randn(4), idx); \
e = knn_graph(torch.randn(64, 3), k=8); \
assert e.shape == (2, 64 * 8), e.shape; \
print('>>> kernels ok: torch', torch.__version__, torch.version.cuda, '| pyg', torch_geometric.__version__)"

# The graph cache MapDiff trains off is pickled torch_geometric Data objects,
# and torch 2.6 flipped torch.load's weights_only default to True, which
# refuses to unpickle them. dataloader/pyg_safe_globals.py allow-lists the
# classes involved; prove those allow-list entries still resolve in the
# installed pyg, since a rename upstream would only surface at data-load time.
RUN python -c "\
import torch; \
from torch_geometric.data.data import Data, DataEdgeAttr, DataTensorAttr; \
from torch_geometric.data.storage import GlobalStorage; \
torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage]); \
import io; \
buf = io.BytesIO(); torch.save(Data(x=torch.randn(4, 3)), buf); buf.seek(0); \
assert torch.load(buf).x.shape == (4, 3); \
print('>>> pyg pickle round-trip ok under weights_only default')"

WORKDIR /workspace

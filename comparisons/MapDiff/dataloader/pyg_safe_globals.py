"""Let `torch.load` read our featurised graph cache on torch >= 2.6.

torch 2.6 flipped `torch.load`'s `weights_only` default to True, which unpickles
only tensors and plain containers -- so every `torch.load` of a
`surffold_data/<ds>_process/**/<protein>.pt` graph (dataloader/utils.py,
dataloader/large_dataset.py, data/generate_graph_relaxed.py) now fails with
"Weights only load failed ... GLOBAL torch_geometric.data.data.DataEdgeAttr was
not an allowed global".

These files are written by data/generate_graph_relaxed.py in this same pipeline,
so the fix is to allow-list the PyG classes they contain rather than turning
`weights_only` off wholesale -- checkpoints pulled from elsewhere (esmfold_v1 in
lib/scrmsd.py) keep the safe default.
"""
import torch
from torch_geometric.data.data import Data, DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage


def allow_pyg_data_pickles():
    torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage])

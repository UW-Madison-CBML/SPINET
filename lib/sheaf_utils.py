import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial import distance_matrix
from torch_geometric.data import Data
from torch_geometric.utils import sort_edge_index
from torch_geometric.nn import MessagePassing

class MessageUtil(MessagePassing):
    def __init__(self):
        super().__init__(aggr='sum', node_dim=0)
    def forward(self, maps, edge_index):
        return self.propagate(edge_index, maps=maps)
    def message(self, maps):
        return maps

    

def sheaf_laplacian(n, maps, edge_index):
    _, d, _ = maps.shape
    assert maps.shape[1] == maps.shape[2], "sheaf maps are not square"
    diagonal_maps = torch.matmul(maps.mT, maps) # diag_i: sum_{i ~ j} F^T_{i <= ij} @ F_{i <= ij}
    agged_maps = MessageUtil()(diagonal_maps, edge_index) 
    sheaf_laplacian_on_diag = torch.block_diag(*agged_maps) 
    sheaf_laplacian_off_diag = torch.zeros((n, n, d, d))
    _, opposite_indices = sort_edge_index(edge_index.roll(0,1), edge_attr=torch.arange(edge_index.shape[1]))
    opp_maps = maps[opposite_indices]
    sheaf_laplacian_off_diag[edge_index[0], edge_index[1]] = torch.matmul(opp_maps.mT, maps)
    sheaf_laplacian = sheaf_laplacian_on_diag + sheaf_laplacian_off_diag.permute(0,2,1,3).reshape(n*d, n*d)
    return sheaf_laplacian

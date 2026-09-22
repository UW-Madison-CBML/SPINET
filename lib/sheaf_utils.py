import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial import distance_matrix
from torch_geometric.data import Data
from torch_geometric.utils import sort_edge_index
from torch_geometric.nn import MessagePassing


    

def sheaf_laplacian(n, maps, edge_index):
    _, d, _ = maps.shape
    assert maps.shape[1] == maps.shape[2], "sheaf maps are not square"
    e = edge_index.shape[1]
    _, opposite_edge_indices = sort_edge_index(torch.roll(egde_index,1,0), edge_attr=torch.arange(e, device=edge_index.device))
    cobound = torch.zeros(e, n, d, d, device=maps.device)
    cobound[torch.arange(e), edge_index[0]] = maps
    cobound[torch.arange(e), edge_index[1]] = -1 * maps[opposite_edge_indices]
    cobound = cobound.permute(0,2,1,3).reshape(e*d, n*d)
    return cobound.T @ cobound

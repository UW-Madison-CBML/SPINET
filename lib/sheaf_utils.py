import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial import distance_matrix
from torch_geometric.data import Data

   
    

def sheaf_laplacians(n, maps, edge_index):
    _, d, _ = maps.shape
    
    assert maps.shape[1] == maps.shape[2], "sheaf maps are not square"

    return torch.zeros((n*d,n*d))

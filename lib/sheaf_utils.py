import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial import distance_matrix
from torch_geometric.nn import radius_graph
from torch_geometric.data import Data

def build_graph(coords, feats, labels, mask, epsilon=5.0, add_temporal_edges=True):
     
    pos = torch.as_tensor(coords, dtype=torch.float32)
    x = torch.as_tensor(feats, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.long)
    node_mask = torch.as_tensor(mask, dtype=torch.bool)
 
    edge_index = radius_graph(pos, r=epsilon, max_num_neighbors=32, loop=False)
    edge_attr = (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1, keepdim=True)
 
    if add_temporal_edges and pos.size(0) > 1:
        idx = torch.arange(pos.size(0) - 1)
        temporal = torch.stack([
            torch.cat([idx, idx + 1]),
            torch.cat([idx + 1, idx]),
        ])
        temporal_attr = (pos[temporal[0]] - pos[temporal[1]]).norm(dim=-1, keepdim=True)
        edge_index = torch.cat([edge_index, temporal], dim=1)
 
    return Data(x=x, pos=pos, y=y, node_mask=node_mask, edge_index=edge_index)
    
    

def eigenspectrum(laplacians, lengths):
    
    # view the two laplacians together
    B, TD, _ = laplacians.shape
    assert lengths.shape == (B,), f"WRONG SHAPE: {lengths.shape}"
    assert lengths.max() <= TD and lengths.min() >= 1, f"BAD RANGE FOR LENGTHS: expected: [{lengths.min()},{lengths.max()}] is not a subset of [1, {TD}]"

    padding = (torch.arange(TD, device=laplacians.device)[None,:] < lengths[:, None])
    identity_mask = padding[:,None,:] & padding[:,:,None]
    
    # need to pad the laplacians with identity columns giving us extra -1 eigvals = T - padding
    identity = -1 * torch.eye(TD, dtype=torch.bool, device = laplacians.device) 
    identity = identity.reshape((1, TD, TD))
    identity = identity.repeat(B, 1, 1)

    masked_laplacians = torch.where(identity_mask, laplacians, identity) # sheaf laplacians always has positive eigenvalues, if we pad with negative identity, we know the -1 eignvalues cannot belong to the sheaf laplacian
    
    # if our laplacians are symmetric we can use eigvalsh
    # otherwise just use eigvals
    print(masked_laplacians.shape)
    eigenspectra = torch.linalg.eigvals(masked_laplacians)
    return eigenspectra

def eigenvectors(laplacians, lengths):
    """Outputs eigenvectors for laplacians as n x k matrix M."""
    # view the two laplacians together
    B, TD, _ = laplacians.shape
    assert lengths.shape == (B,), f"WRONG SHAPE: {lengths.shape}"
    assert lengths.max() <= TD and lengths.min() >= 1, f"BAD RANGE FOR LENGTHS: expected: [{lengths.min()},{lengths.max()}] is not a subset of [1, {TD}]"

    padding = (torch.arange(TD, device=laplacians.device)[None,:] < lengths[:, None])
    identity_mask = padding[:,None,:] & padding[:,:,None]

    # need to pad the laplacians with identity columns giving us extra -1 eigvals = T - padding
    identity = -1 * torch.eye(TD, dtype=laplacians.dtype, device = laplacians.device) 
    identity = identity.reshape((1, TD, TD))
    identity = identity.repeat(B, 1, 1)


    masked_laplacians = torch.where(identity_mask, laplacians, identity) # sheaf laplacians always has positive eigenvalues, if we pad with negative identity, we know the -1 eignvalues cannot belong to the sheaf laplacian

    # Tikhonov regularization to avoid degeneracy
    eps = 1e-5

    N = masked_laplacians.size(-1)

    # Force noise to float64 for numerical stability
    noise = torch.rand(N, device=masked_laplacians.device, dtype=torch.float64) * eps
    jitter = torch.diag_embed(noise) #create diag matrix from noise vector

    masked_laplacians = masked_laplacians + jitter

    # Enforce symmetry to correct floating-point drift
    masked_laplacians = (masked_laplacians + masked_laplacians.transpose(-1, -2)) / 2.0

    eigvals, eigvecs = torch.linalg.eigh(masked_laplacians)

    return eigvals.to(laplacians.dtype), eigvecs.to(laplacians.dtype)

if __name__ == "__main__":
    B = 2      # Batch size
    TD = 5     # Maximum sequence length (Target Dimension)
    
    # Create dummy laplacians (symmetric positive semi-definite)
    # Shape: (B, TD, TD)
    random_matrices = torch.randn(B, TD, TD)
    laplacians = torch.bmm(random_matrices, random_matrices.transpose(1, 2)) 
    
    # Create variable lengths for the batch
    lengths = torch.tensor([3, 5]) 
    
    # 3. Run the function
    eigvals, eigvecs = eigenvectors(laplacians, lengths)
    
    # 4. Assertions and Checks
    print(f"Eigenvalues shape: {eigvals.shape} (Expected: {B}, {TD})")
    print(f"Eigenvectors shape: {eigvecs.shape} (Expected: {B}, {TD}, {TD})\n")
    
    # Let's check the first item in the batch (length 3, padded to 5)
    # Because it was padded with -1s, the smallest eigenvalues should be exactly -1
    full_eigvals, _ = torch.linalg.eigh(laplacians[0])
    
    # Manually masking the first matrix to show the -1 padding worked
    padding_mask = (torch.arange(TD) < lengths[0]).unsqueeze(1) & (torch.arange(TD) < lengths[0]).unsqueeze(0)
    identity = -1 * torch.eye(TD)
    masked_matrix = torch.where(padding_mask, laplacians[0], identity)
    test_vals, _ = torch.linalg.eigh(masked_matrix)
    
    print("First item in batch (True Length = 3, Padded to = 5):")
    print(f"Eigenvalues after padding mask: {test_vals.round(decimals=3).tolist()}")

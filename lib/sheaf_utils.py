import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial import distance_matrix

def build_graph(conformations1, conformations2, lengths, epsilon, adjacency_matrix=True):
    """
    Creates edges between nodes (atoms/residues) epsilon distance from one another.
    Processes two conformations at once so they are transformed in same way for valid comparison.
    Outputs either an adjacency matrix or a dense batched edge list.

    Args:
        conformations1: Batched 3-D tensor of residue positions for conf 1 (B, T, 3).
        conformations2: Batched 3-D tensor of residue positions for conf 2 (B, T, 3).
        lengths: Number of valid residues per batch (B,).
        epsilon: Max distance for edge between residues.
        adjacency_matrix: If True, returns bool matrix. If False, returns batched edge lists.
    Return:
        If adjacency_matrix=True:
            adjacency: (B, 2, T, T) boolean tensor
        If adjacency_matrix=False:
            out_list: (B, 2, E_max, 2) tensor of edges, padded with (0,0)
            out_lengths: (B, 2) tensor of valid edge counts
    """
    B, T, _ = conformations1.shape
    
    # padding needs to be on CUDA
    padding = (torch.arange(T, device=conformations1.device)[None, :] < lengths[:, None])

    dist_mat1 = torch.cdist(conformations1, conformations1, p=2) # B, T, T
    dist_mat2 = torch.cdist(conformations2, conformations2, p=2) # B, T, T
    dist_mat = torch.stack([dist_mat1, dist_mat2], dim=1)        # B, 2, T, T

    matrix_padding = padding[:, None, None, :] & padding[:, None, :, None] # B, 1, T, T
    
    adjacency = (dist_mat < epsilon) & matrix_padding

    if adjacency_matrix:
        # Remove self edges (diagonal) for the adjacency matrix
        return adjacency & (~torch.eye(T, dtype=torch.bool, device=adjacency.device)[None, None, :, :])

    # For edge list:
    # Remove self-loops (diagonal=1) and symmetric duplicates by keeping only upper triangle
    triu_mask = torch.triu(torch.ones((T, T), dtype=torch.bool, device=adjacency.device), diagonal=1)
    adjacency = adjacency & triu_mask[None, None, :, :]

    # Flatten spatial dimensions to process edge extraction
    adj_flat = adjacency.view(B, 2, T * T)

    # Find the maximum number of valid edges across the entire batch
    E = adj_flat.sum(dim=2).max().item()
    
    # Handle edge case where no nodes are within epsilon
    if E == 0:
        return torch.zeros((B, 2, 0, 2), dtype=torch.long, device=adjacency.device), \
               torch.zeros((B, 2), dtype=torch.long, device=adjacency.device)

    # Sort pushes True (1) values to the front. 
    # This neatly organizes all valid edges to the start of the list.
    _, indices = torch.sort(adj_flat.int(), dim=2, descending=True)
    indices = indices[:, :, :E] # [B, 2, E]

    # Convert 1D flat indices back to 2D (row, col) coordinates
    rows = torch.div(indices, T, rounding_mode='floor')
    cols = indices % T
    
    # Stack into [B, 2, E, 2]
    edges = torch.stack([rows, cols], dim=-1)

    # We need to mask out the dummy edges that got pulled in by the fixed `E` dimension.
    # Check if the pulled indices were actually True in the original adjacency matrix.
    valid_edge_mask = torch.gather(adj_flat, dim=2, index=indices).unsqueeze(-1) # [B, 2, E, 1]

    # PyG will CRASH if given a negative index like -1. 
    # By multiplying by the boolean mask, padded invalid edges safely default to (0, 0).
    out_list = edges * valid_edge_mask

    # Get the actual number of valid edges per graph
    out_lengths = valid_edge_mask.sum(dim=2).squeeze(-1) # [B, 2]

    return out_list, out_lengths 
     
    
    

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

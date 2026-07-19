import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial import distance_matrix

def build_graph(conformations1, conformations2, lengths, epsilon, adjacency_matrix=True):
    """
    Creates edges between nodes (atoms/residues) epsilon distance from one another.
    Processes two conformations at once so they are transformed in same way for valid comparison.
    Outputs adjacency list. 

    Args:
        conformations1: Batched 3-D tensor of residue positions for conf 1.
        conformations2: Batched 3-D tensor of residue positions for conf 2.
        padding: Dynamic padding mask.
        epsilon: Learned max distance for edge between residues.
    Return:
        out_list: Stacked adjacency lists.
        out_padding: Nested lists False if node is padded, True otherwise.

    """
    # B, T, 3 = conformations1.shape = conformations2.shape
    B,T, _ = conformations1.shape
    assert lengths.shape == (B,), f"WRONG SHAPE: {lengths.shape}"
    assert lengths.max() <= T and lengths.min() >= 1, f"BAD RANGE FOR LENGTHS: expected: [{lengths.min()},{lengths.max()}] is not a subset of [1, {T}]"
    # lengths = (B), max < T
    # padding needs to be on CUDA because of (1)
    # padding = (B,T)
    padding = (torch.arange(T)[None,:] < lengths[:, None]).to(conformations1.device)

    dist_mat1 = torch.cdist(conformations1, conformations1, p=2) # B, T, T
    dist_mat2 = torch.cdist(conformations2, conformations2, p=2) # B, T, T
    dist_mat = torch.stack([dist_mat1, dist_mat2], dim=1)
    # TODO implement some other form of predicate to determine existence of edges
    print("dist_mat.shape", dist_mat.shape)
    matrix_padding = padding[:,None, None, :] & padding[:,None, :, None] # B, 1, T, T
    assert matrix_padding.shape == (B, 1, T, T), f"WRONG SHAPE: {matrix_padding.shape}" 
    print("matrix_padding.shape: ", matrix_padding.shape)
    print("epsilon.shape: ", epsilon.shape)
    adjacency = (dist_mat < epsilon) & matrix_padding # (1)
    print("adj.shape:", adjacency.shape)

    # Remove self-loops and symmetric duplicates by keeping only upper triangle
    # The data structure I'm using in the sheaf laplacian is edges = (B,E,2) a set 
    # of unique edges whose indices are < T, and >= 0. Padding is done with the pairs (-1, -1). 
    # The sheaves or sets of restriction maps are of shape (B,E,2,D,D) where at index [:,i] we have 
    # the 2 restriction maps from node edges[:,i,0] to edge edges[:,i] and from node edges[:,i,1] to edges[:,i]
    # clear out redundant edges 
    if not adjacency_matrix:
        triu_mask = torch.triu(torch.ones((T,T), dtype=torch.bool, device=adjacency.device), diagonal=0)
        adjacency = adjacency & triu_mask[None, None, :, :]

    #if we index restriction maps via adjacency mats 
    # remove the diagonal, no self edges
    else:
        return adjacency & (~torch.eye(T, dtype=torch.bool, device=adjacency.device)[None,None,:,:])

    #alternatively if we want to do the edges list 
    rows = torch.arange(T)[None,None,:,None].repeat(B,2, 1, 1)
    cols = torch.arange(T)[None,None,None,:].repeat(B,2, 1, 1)
    rows, cols = torch.broadcast_tensors(rows, cols) # B, 2, T, T
    
    edges = torch.stack([rows, cols], dim=-1) 
    edges = torch.where(padding[:,None, None, :,None] & padding[:, None, :, None,None], edges, -1)

    # Flatten
    adjacency = adjacency.reshape(B,2,T*T)
    edges = edges.reshape(B,2, T*T,2)

    # Max edges per graph rather than sum across both
    E = adjacency.sum(dim=2).max().to(int).item() 

    # Mask invalid edges before topk
    edges = torch.where(adjacency.unsqueeze(-1), edges, -1)

    # Push valid edges to front and get indices
    _, indices = torch.topk(adjacency.to(int),E,dim=2) # [B,2,E] 

    # Get indices with dummy dimensions
    b_idx = torch.arange(B)[:, None, None] # [B, 1, 1]
    c_idx = torch.arange(2)[None, :, None] # [1, 2, 1]

    # Gather edges
    out_list = edges[b_idx, c_idx, indices] # [B,2, E, 2]
    # out list should be equal along the pair dims
    out_lengths = ((out_list[:,0,:,0] == -1) | (out_list[:,0,:,1] == -1)).sum(dim=-1) # B, max < E
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

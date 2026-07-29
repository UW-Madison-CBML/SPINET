import torch
import torch_sparse

def sym_matrix_pow(matrix: torch.Tensor, p: float) -> torch.Tensor:
    """
    Batched and differentiable power of a matrix using Eigen Decomposition.
    Args:
        matrix: tensor of shape (..., K, K)
        p: power
    """
    vals, vecs = torch.linalg.eigh(matrix)
    
    # Differentiable, out-of-place thresholding to prevent NaN gradients from negative eigenvalues
    vals_pow = torch.where(vals > 1e-6, vals.pow(p), torch.zeros_like(vals))
    
    # Use diag_embed and .mT to support batched matrices (N, K, K)
    matrix_pow = vecs @ torch.diag_embed(vals_pow) @ vecs.mT
    return matrix_pow


def _build_laplacian_blocks(N, K, edge_index, maps):
    """Computes the raw KxK Laplacian blocks for each edge."""
    E = edge_index.size(1)

    # Phi for source (u) and target (v)
    Phi_u = maps[:, 0, :, :]  # Shape: (E, K, K)
    Phi_v = maps[:, 1, :, :]  # Shape: (E, K, K)

    # Compute the 4 blocks for each edge using batched matrix multiplication (fully differentiable)
    # L_uu = Phi_u * Phi_u^T
    L_uu = torch.bmm(Phi_u, Phi_u.transpose(1, 2))
    # L_vv = Phi_v * Phi_v^T
    L_vv = torch.bmm(Phi_v, Phi_v.transpose(1, 2))
    # L_uv = -Phi_u * Phi_v^T
    L_uv = -torch.bmm(Phi_u, Phi_v.transpose(1, 2))
    # L_vu = -Phi_v * Phi_u^T
    L_vu = -torch.bmm(Phi_v, Phi_u.transpose(1, 2))

    return L_uu, L_vv, L_uv, L_vu


def _scatter_blocks_to_sparse(N, K, edge_index, L_uu, L_vv, L_uv, L_vu):
    """Scatters the batched blocks into a cohesive (index, value) sparse format."""
    E = edge_index.size(1)
    device = L_uu.device
    u, v = edge_index[0], edge_index[1]

    # Precompute KxK grid indices
    k_row = torch.arange(K, device=device).view(K, 1).expand(K, K).flatten()
    k_col = torch.arange(K, device=device).view(1, K).expand(K, K).flatten()

    # Map grid indices to the absolute node positions in the N*K x N*K matrix
    row_uu = (u.view(E, 1) * K + k_row.view(1, K*K)).flatten()
    col_uu = (u.view(E, 1) * K + k_col.view(1, K*K)).flatten()

    row_vv = (v.view(E, 1) * K + k_row.view(1, K*K)).flatten()
    col_vv = (v.view(E, 1) * K + k_col.view(1, K*K)).flatten()

    row_uv = (u.view(E, 1) * K + k_row.view(1, K*K)).flatten()
    col_uv = (v.view(E, 1) * K + k_col.view(1, K*K)).flatten()

    row_vu = (v.view(E, 1) * K + k_row.view(1, K*K)).flatten()
    col_vu = (u.view(E, 1) * K + k_col.view(1, K*K)).flatten()

    # Concatenate all indices and values
    L_row = torch.cat([row_uu, row_vv, row_uv, row_vu])
    L_col = torch.cat([col_uu, col_vv, col_uv, col_vu])
    L_val = torch.cat([L_uu.flatten(), L_vv.flatten(), L_uv.flatten(), L_vu.flatten()])

    L_idx = torch.stack([L_row, L_col], dim=0)

    # coalesce() naturally sums duplicate indices (e.g. L_uu blocks intersecting on the diagonal)
    return torch_sparse.coalesce(L_idx, L_val, N * K, N * K)


def build_sheaf_laplacian(N, K, edge_index, maps):
    """Builds a regular sheaf laplacian directly."""
    L_uu, L_vv, L_uv, L_vu = _build_laplacian_blocks(N, K, edge_index, maps)
    return _scatter_blocks_to_sparse(N, K, edge_index, L_uu, L_vv, L_uv, L_vu)


def build_norm_sheaf_laplacian(N, K, edge_index, maps, augmented=True):
    """
    Builds a normalized sheaf laplacian cleanly by normalizing the blocks
    before assembling the sparse matrix, bypassing spspmm gradient destruction.
    """
    L_uu, L_vv, L_uv, L_vu = _build_laplacian_blocks(N, K, edge_index, maps)
    
    device = maps.device
    u, v = edge_index[0], edge_index[1]

    # 1. Compute Degree Matrix (D) blocks simultaneously using index_add
    D = torch.zeros((N, K, K), device=device, dtype=maps.dtype)
    D.index_add_(0, u, L_uu)
    D.index_add_(0, v, L_vv)

    if augmented:
        D = D + torch.eye(K, device=device).unsqueeze(0)

    # 2. Compute D^{-1/2} for all N nodes in one shot (Batched)
    D_inv_sqrt = sym_matrix_pow(D, -0.5)

    # 3. Fetch the D^{-1/2} block for each edge's source and target
    Du = D_inv_sqrt[u] # Shape: (E, K, K)
    Dv = D_inv_sqrt[v] # Shape: (E, K, K)

    # 4. Normalize the blocks: L_{uv}^{norm} = D_u^{-1/2} L_{uv} D_v^{-1/2}
    L_uu_norm = torch.bmm(torch.bmm(Du, L_uu), Du)
    L_vv_norm = torch.bmm(torch.bmm(Dv, L_vv), Dv)
    L_uv_norm = torch.bmm(torch.bmm(Du, L_uv), Dv)
    L_vu_norm = torch.bmm(torch.bmm(Dv, L_vu), Du)

    # 5. Scatter to sparse matrix
    return _scatter_blocks_to_sparse(N, K, edge_index, L_uu_norm, L_vv_norm, L_uv_norm, L_vu_norm)

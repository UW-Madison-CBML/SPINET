import os
import torch
import numpy as np
from typing import List, Tuple, Dict

import torch
from transformers import AutoTokenizer, EsmForProteinFolding

BACKBONE_ATOM14_IDX = {"N":0, "CA":1, "C":2, "O":4}  # N, CA, C, O # TODO fix this
from load_dynamics import BACKBONE_ATOMS

@torch.no_grad()
def kabsch_rmsd(P: torch.Tensor, Q: torch.Tensor, mask: torch.Tensor = None, device="cpu"):
    """
    P: b_1, ..., b_n, num_atoms, 3  # make sure P and Q are flat in that -2 dim, since there is no residue structre to uphold, just atoms
    Q: b_1, ..., b_n, num_atoms, 3 
    mask: b_1, ..., b_n, num_atoms, type=bool
    device: if you wanna use GPU to do a big batch you can
    returns: 
        rmsd: b_1, ..., b_n 
    """
    P = P.to(device)
    Q = Q.to(device)
    assert Q.shape == P.shape, "P and Q do not have same shape"
    n = P.shape[-1]
    if mask is not None:
        assert mask.any(dim=-1).all().item(), "each molecule must have at least one atom"

        # einsum's CUDA path has no bool kernel ("baddbmm_cuda" not implemented for 'Bool'),
        # and callers pass a bool validity mask -- match P's dtype up front.
        mask = mask.to(device=device, dtype=P.dtype)

        mask_extra_dim = mask.unsqueeze(-1) # so that it broadcasts

        # center it 
        # this will not fail because we check that each molecule has at least one atom, thus mask.sum(dim=-1) >= 1
        P = P - ((P * mask_extra_dim).sum(dim=-2, keepdim=True) / mask_extra_dim.sum(-2, keepdim=True))
        Q = Q - ((Q * mask_extra_dim).sum(dim=-2, keepdim=True) / mask_extra_dim.sum(-2, keepdim=True))


        # calculate covariance
        H = torch.einsum("...si,...sj,...s->...ij", P, Q, mask)
        
        # SVD
        U, _, Vt = torch.linalg.svd(H)

        # determinant
        d = torch.linalg.det(U) * torch.linalg.det(Vt)

        # this puts 1, ,...,  1, d on the diagonal
        d_diag = torch.diag_embed(torch.cat([torch.ones(*d.shape, n-1, device=d.device), d.unsqueeze(-1)], dim=-1))
        
        # calculate R
        R = torch.matmul(torch.matmul(U, d_diag), Vt)
       
        Q_rotated = torch.matmul(Q, R.mT)

        # calculate RMSD
        errors = torch.linalg.norm(P - Q_rotated, dim=-1)
        squared_errors = errors ** 2
        mean_squared_errors = (squared_errors * mask).sum(dim=-1) / mask.sum(dim=-1) 
        rmsd = torch.sqrt(mean_squared_errors) 
     

    else:
        # center it 
        P = P - P.mean(dim=-2, keepdim=True)
        Q = Q - Q.mean(dim=-2, keepdim=True)

        # calculate covariance
        H = torch.einsum("...si,...sj->...ij", P, Q)
        
        # SVD
        U, _, Vt = torch.linalg.svd(H)

        # determinant
        d = torch.linalg.det(U) * torch.linalg.det(Vt)

        # this puts 1, 1, d on the diagonal
        d_diag = torch.diag_embed(torch.cat([torch.ones(*d.shape, n-1, device=d.device), d.unsqueeze(-1)], dim=-1)) # 3,3
        
        # calculate R
        R = torch.matmul(torch.matmul(U, d_diag), Vt)
       
        Q_rotated = torch.matmul(Q, R.mT)

        # calculate RMSD
        errors = torch.linalg.norm(P - Q_rotated, dim=-1)
        squared_errors = errors ** 2
        mean_squared_errors = squared_errors.mean(dim=-1)
        rmsd = torch.sqrt(mean_squared_errors) 
     
    return rmsd.cpu()
 

def load_esmfold(
    model_name: str = "facebook/esmfold_v1",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: torch.dtype = torch.float32,
    chunk_size: int = 64,
):
    """`chunk_size` chunks ESMFold's triangular attention/multiplication along the residue axis.

    The folding trunk's attention logits are ~B*heads*L^3 floats, so a single ~1100-residue
    protein asks for a 44 GiB softmax and OOMs even a 140 GiB GPU. Chunking trades a modest
    slowdown for a large drop in peak memory; `None` restores the unchunked default.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmForProteinFolding.from_pretrained(model_name, low_cpu_mem_usage=True)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    trunk = getattr(model, "trunk", None)
    if chunk_size is not None and hasattr(trunk, "set_chunk_size"):
        trunk.set_chunk_size(chunk_size)

    return tokenizer, model


@torch.no_grad()
def fold_sequences(
    seqs: list[str],
    tokenizer,
    model,
    device: str = None,
):
    
    device = torch.device(device) if device is not None else next(model.parameters()).device
    bb_idx = [BACKBONE_ATOM14_IDX[atom] for atom in BACKBONE_ATOMS]

    # Folded one at a time, then padded back to the batch's longest. Folding the padded batch
    # in one call costs B * L_max^3 in the trunk's attention -- a short sequence batched with a
    # long one gets folded at the long one's length. Peak memory here is that of the single
    # longest sequence instead, and the (B, L_max, ...) return contract is unchanged.
    per_seq = []
    for seq in seqs:
        inputs = tokenizer(
            [seq],
            return_tensors="pt",
            padding=False,
            add_special_tokens=False,  # ESMFold doesn't use BOS/EOS
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)

        positions = outputs.positions
        if positions.dim() == 6:
            positions = positions[-1]
        atom14 = positions[-1]  # (1, L, 14, 3) — final structure-module layer

        per_seq.append((atom14[:, :, bb_idx, :].float().squeeze(0),      # (L, 4, 3)
                        inputs["attention_mask"].bool().squeeze(0)))     # (L,)
        del outputs, positions, atom14, inputs
        if device.type == "cuda":
            # the trunk's transient pair tensors fragment the caching allocator badly enough
            # that the next, longer sequence can OOM on otherwise-free memory
            torch.cuda.empty_cache()

    L_max = max(int(m.shape[0]) for _, m in per_seq)
    backbone_pos = torch.zeros(len(per_seq), L_max, len(bb_idx), 3,
                               dtype=per_seq[0][0].dtype, device=device)
    mask = torch.zeros(len(per_seq), L_max, dtype=torch.bool, device=device)
    for i, (bb, m) in enumerate(per_seq):
        backbone_pos[i, : bb.shape[0]] = bb
        mask[i, : m.shape[0]] = m

    return backbone_pos, mask
 
   



def evaluate_batch_rmsd(
    sequences: List[str], 
    ground_truth_coords: torch.Tensor,
    gt_mask: torch.Tensor,
    tokenizer,
    model,
    device="cpu"
):
    pred_coords, pred_mask = fold_sequences(sequences, tokenizer, model, device=device)

    B,    R, A, _ = pred_coords.shape

    _, T, _, _, _ = ground_truth_coords.shape
    ground_truth_coords = ground_truth_coords.to(device)
    gt_mask = gt_mask.to(device)

    P_backbone = pred_coords.reshape(B, R*A, 3)[:, None, :, :].expand(-1, T, -1, -1)
    Q_backbone = ground_truth_coords.reshape(B, T, R*A, 3)

    backbone_mask = gt_mask[:,:,:,None].expand(-1, -1, -1, A).reshape(B, T, R*A)

    return kabsch_rmsd(P_backbone, Q_backbone, mask=backbone_mask, device=device).cpu()

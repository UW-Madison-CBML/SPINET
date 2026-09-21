import os
import torch
import numpy as np
from typing import List, Tuple, Dict

import torch
from transformers import AutoTokenizer, EsmForProteinFolding

BACKBONE_ATOM14_IDX = {"N":0, "CA":1, "C":2, "O":4}  # N, CA, C, O # TODO fix this
# The repo-wide ground-truth atom order, re-exported by `relaxed_pdb` rather than imported
# from `load_dynamics`: the comparison models all ship this file, and not all of their images
# carry mdtraj (which `load_dynamics` pulls in at import time).
from relaxed_pdb import BACKBONE_ATOMS
from tqdm import tqdm

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
    assert Q.shape == P.shape, f"P and Q do not have same shape: {Q.shape} {P.shape}"

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

RESTYPE_ATOM14_NAMES = {
    'A': ['N', 'CA', 'C', 'O', 'CB', '', '', '', '', '', '', '', '', ''],
    'R': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2', '', '', ''],
    'N': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2', '', '', '', '', '', ''],
    'D': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2', '', '', '', '', '', ''],
    'C': ['N', 'CA', 'C', 'O', 'CB', 'SG', '', '', '', '', '', '', '', ''],
    'Q': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2', '', '', '', '', ''],
    'E': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2', '', '', '', '', ''],
    'G': ['N', 'CA', 'C', 'O', '', '', '', '', '', '', '', '', '', ''],
    'H': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2', '', '', '', ''],
    'I': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', '', '', '', '', '', ''],
    'L': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', '', '', '', '', '', ''],
    'K': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ', '', '', '', '', ''],
    'M': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE', '', '', '', '', '', ''],
    'F': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', '', '', ''],
    'P': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', '', '', '', '', '', '', ''],
    'S': ['N', 'CA', 'C', 'O', 'CB', 'OG', '', '', '', '', '', '', '', ''],
    'T': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', '', '', '', '', '', '', ''],
    'W': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    'Y': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH', '', ''],
    'V': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', '', '', '', '', '', '', ''],
}

AA3 = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
}


def atom14_to_pdb(atom14, mask, seq, out_path, chain="A", atom14_atom_exists=None):
    coords = atom14.detach().cpu().numpy() if hasattr(atom14, "detach") else np.asarray(atom14)
    mask = mask.detach().cpu().numpy() if hasattr(mask, "detach") else np.asarray(mask)
    if atom14_atom_exists is not None:
        exists = atom14_atom_exists.detach().cpu().numpy() if hasattr(atom14_atom_exists, "detach") \
            else np.asarray(atom14_atom_exists)

    lines = []
    atom_num = 1
    for res_idx, (res_coords, valid, aa) in enumerate(zip(coords, mask, seq), start=1):
        if not valid:
            continue
        aa_u = aa.upper()
        resname = AA3.get(aa_u, "UNK")
        atom_names = RESTYPE_ATOM14_NAMES.get(aa_u, [""] * 14)

        for slot, (atom_name, (x, y, z)) in enumerate(zip(atom_names, res_coords)):
            if atom14_atom_exists is not None:
                if not exists[res_idx - 1, slot]:
                    continue
            elif atom_name == "":
                continue
            lines.append(
                f"ATOM  {atom_num:5d}  {atom_name:<3s}{resname:>3s} {chain}{res_idx:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{0.00:6.2f}          "
                f"{atom_name[0]:>2s}"
            )
            atom_num += 1
    lines.append("TER")
    lines.append("END")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


   

@torch.no_grad()
def fold_sequences(
    seqs: list[str],
    tokenizer,
    model,
    device: str = None,
    max_tokens_per_batch: int = 2048, # VRAM is O(n**3) here so be careful
    save_pdbs=None # list of paths if not None
):
    device = torch.device(device) if device is not None else next(model.parameters()).device
    bb_idx = [BACKBONE_ATOM14_IDX[atom] for atom in BACKBONE_ATOMS]

    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))

    batches = []
    current = []
    for i in order:
        candidate = current + [i]
        L = len(seqs[max(candidate, key=lambda j: len(seqs[j]))])
        cost = (L ** 3) * len(candidate)
        if current and cost > max_tokens_per_batch ** 3 // 64:  # rough knob, see note below
            batches.append(current)
            current = [i]
        else:
            current = candidate
    if current:
        batches.append(current)

    per_seq = [None] * len(seqs)
    for batch_idx in tqdm(batches):
        batch_seqs = [seqs[i] for i in batch_idx]
        if save_pdbs is not None:
            batch_save_pdbs = [save_pdbs[i] for i in batch_idx]
        inputs = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)

        positions = outputs.positions
        if positions.dim() == 6:
            positions = positions[-1]
        atom14 = positions[-1]  # (B, L, 14, 3)
        if save_pdbs is not None:
            for mol_idx in range(atom14.shape[0]):
                atom14_to_pdb(
                    atom14[mol_idx],                          # (L, 14, 3)
                    inputs["attention_mask"][mol_idx].bool(),
                    batch_seqs[mol_idx],
                    batch_save_pdbs[mol_idx], 
                    atom14_atom_exists=outputs.atom14_atom_exists[mol_idx] if hasattr(outputs, "atom14_atom_exists") else None,
                )
        bb = atom14[:, :, bb_idx, :].float()          # (B, L, 4, 3)
        m = inputs["attention_mask"].bool()            # (B, L)

        for row, orig_i in enumerate(batch_idx):
            L_i = int(m[row].sum())
            per_seq[orig_i] = (bb[row, :L_i], m[row, :L_i])

        del outputs, positions, atom14, inputs, bb, m
        if device.type == "cuda":
            torch.cuda.empty_cache()

    L_max = max(int(m.shape[0]) for _, m in per_seq)
    backbone_pos = torch.zeros(len(per_seq), L_max, len(bb_idx), 3,
                               dtype=per_seq[0][0].dtype, device=device)
    mask = torch.zeros(len(per_seq), L_max, dtype=torch.bool, device=device)
    for i, (bb, m) in enumerate(per_seq):
        backbone_pos[i, : bb.shape[0]] = bb
        mask[i, : m.shape[0]] = m

    return backbone_pos, mask

@torch.no_grad()
def evaluate_scrmsd(
    sequences: List[str],
    reference_coords: List[torch.Tensor],
    tokenizer,
    model,
    pred_indices: List = None,
    ref_indices: List = None,
    device="cpu",
):
    """Self-consistency RMSD of each designed sequence against a single reference structure.

    This is the scRMSD every comparison model reports: fold the predicted sequence with
    ESMFold and Kabsch-RMSD the folded backbone against the protein's ground-truth *relaxed*
    (deposited) structure -- see lib/relaxed_pdb.py.

    sequences: length-B list of 1-letter amino-acid strings, one design per protein.
    reference_coords: length-B list of (R_ref_i, A, 3) ground-truth backbone coordinates.
    pred_indices / ref_indices: optional length-B lists of equal-length index arrays giving
        the residue correspondence between the folded prediction and the reference. Needed
        when the two are not positionally identical -- DynamicMPNN designs over the
        trajectory's residues, whose deposited structure resolves a different (aligned)
        subset. Default (None) pairs residue i with residue i, which requires the sequence
        and the reference to have the same length.

    Returns a (B,) cpu tensor of RMSDs, in whatever units `reference_coords` is in
    (Angstroms for lib/relaxed_pdb records, since deposited PDBs are in Angstroms).
    """
    if len(sequences) != len(reference_coords):
        raise ValueError("{} sequences vs {} reference structures".format(
            len(sequences), len(reference_coords)))

    pred_coords, _ = fold_sequences(sequences, tokenizer, model, device=device)  # (B, L_max, A, 3)
    B, _, num_atoms, _ = pred_coords.shape

    selected_pred, selected_ref = [], []
    for i, seq in enumerate(sequences):
        ref = torch.as_tensor(reference_coords[i], dtype=pred_coords.dtype, device=pred_coords.device)
        if pred_indices is None:
            if ref.shape[0] != len(seq):
                raise ValueError(
                    "protein {}: {}-residue reference vs {}-residue design; pass "
                    "pred_indices/ref_indices to score an aligned subset".format(i, ref.shape[0], len(seq)))
            p_idx = torch.arange(len(seq), device=pred_coords.device)
            r_idx = p_idx
        else:
            p_idx = torch.as_tensor(pred_indices[i], dtype=torch.long, device=pred_coords.device)
            r_idx = torch.as_tensor(ref_indices[i], dtype=torch.long, device=pred_coords.device)
        selected_pred.append(pred_coords[i, p_idx])
        selected_ref.append(ref[r_idx])

    # Pad to the longest correspondence and mask the padding out of the superposition, so the
    # whole batch superposes in one shot.
    lengths = torch.tensor([p.shape[0] for p in selected_pred])
    if (lengths == 0).any():
        raise ValueError("at least one protein has no aligned residues to score")
    pad_size = int(lengths.max().item())

    P = torch.zeros(B, pad_size, num_atoms, 3, dtype=pred_coords.dtype, device=pred_coords.device)
    Q = torch.zeros_like(P)
    for i, (p, q) in enumerate(zip(selected_pred, selected_ref)):
        P[i, : p.shape[0]] = p
        Q[i, : q.shape[0]] = q

    residue_mask = lengths[:, None].to(pred_coords.device) > torch.arange(pad_size, device=pred_coords.device)[None, :]
    atom_mask = residue_mask[:, :, None].expand(-1, -1, num_atoms).reshape(B, pad_size * num_atoms)

    return kabsch_rmsd(P.reshape(B, pad_size * num_atoms, 3),
                       Q.reshape(B, pad_size * num_atoms, 3),
                       mask=atom_mask, device=device)
def evaluate_batch_rmsd(
    sequences: List[str],
    ground_truth_coords: torch.Tensor,
    gt_mask: torch.Tensor,
    tokenizer,
    model,
    device="cpu",
    save_pdbs = None
):
    pred_coords, pred_mask = fold_sequences(sequences, tokenizer, model, device=device, save_pdbs=save_pdbs)

    B, R1, A, _ = ground_truth_coords.shape
    assert gt_mask.shape == (B, R1)

    B, R2, A, _ = pred_coords.shape
    R = min(R1, R2)
    pred_coords = pred_coords[:,:R,:,:]
    ground_truth_coords = ground_truth_coords[:,:R,:,:]
    gt_mask = gt_mask[:,:R, None].expand(-1,-1, A)

    ground_truth_coords = ground_truth_coords.reshape(B, R*A, 3)
    pred_coords = pred_coords.reshape(B, R*A, 3)
    gt_mask = gt_mask.reshape(B,R*A)


    ground_truth_coords = ground_truth_coords.to(device)
    gt_mask = gt_mask.to(device)

    P_backbone = pred_coords.to(device)
    Q_backbone = ground_truth_coords

    return kabsch_rmsd(P_backbone, Q_backbone, mask=gt_mask, device=device).cpu()


import os
import torch
import numpy as np
from typing import List, Tuple, Dict
from alphafold.common import residue_constants
from alphafold.data import pipeline
from alphafold.model import config, model

def kabsch_rmsd(P: torch.Tensor, Q: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    lead_shape = P.shape[:-2]
    N = P.shape[-2]
    P = P.reshape(-1, N, 3)
    Q = Q.reshape(-1, N, 3)
    B = P.shape[0]
 
    if mask is None:
        mask = torch.ones(B, N, device=P.device, dtype=P.dtype)
    else:
        mask = mask.reshape(-1, N).to(P.dtype)
 
    counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)          # (B,1)
 
    P_centroid = (P * mask.unsqueeze(-1)).sum(1, keepdim=True) / counts.unsqueeze(-1)
    Q_centroid = (Q * mask.unsqueeze(-1)).sum(1, keepdim=True) / counts.unsqueeze(-1)
 
    P_c = (P - P_centroid) * mask.unsqueeze(-1)
    Q_c = (Q - Q_centroid) * mask.unsqueeze(-1)
 
    H = torch.einsum('bni,bnj->bij', P_c, Q_c)                     # (B,3,3)
    U, _, Vt = torch.linalg.svd(H)
    V = Vt.transpose(-2, -1)
    Ut = U.transpose(-2, -1)
 
    d = torch.sign(torch.linalg.det(torch.einsum('bij,bjk->bik', V, Ut)))
    ones = torch.ones_like(d)
    D = torch.diag_embed(torch.stack([ones, ones, d], dim=-1))     # (B,3,3)
 
    R = torch.einsum('bij,bjk,bkl->bil', V, D, Ut)                 # (B,3,3)
    P_aligned = torch.einsum('bij,bnj->bni', R, P_c)
 
    sq_err = ((P_aligned - Q_c) ** 2).sum(-1) * mask                # (B,N)
    mse = sq_err.sum(1) / counts.squeeze(-1)
    rmsd = torch.sqrt(mse.clamp(min=1e-12))
 
    return rmsd.reshape(lead_shape)
 

 
   

class ColabFoldValidationEngine:
    def __init__(
        self, 
        backbone_atoms:list[str],
        model_name: str = "model_1_ptm", 
        data_dir: str = "./alphafold/data", 
        num_recycles: int = 3,
        device: str = "cuda"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        self.cfg = config.model_config(model_name)
        self.cfg.model.num_recycle = num_recycles
        
        params_path = os.path.join(data_dir, "params", f"params_{model_name}.npz")
        if os.path.exists(params_path):
            import pickle
            with open(params_path, 'rb') as f:
                self.params = pickle.load(f)
        else:
            print(f"Warning: Parameters not found at {params_path}. Using placeholder params for compilation initialization.")
            self.params = None
            
        self.runner = model.RunModel(self.cfg, params=self.params)
        
        self.target_atom_indices = [
            residue_constants.atom_order[atom] for atom in backbone_atoms # this way we have a GT ordering
        ]

    def _generate_single_sequence_features(self, sequence: str) -> dict:
        """
        Generates standard AlphaFold sequence-level feature dictionaries in-memory 
        without spinning up external alignment servers (Zero-Shot / Single-sequence fallback).
        """
        num_res = len(sequence)
        
        sequence_features = pipeline.make_sequence_features(
            sequence=sequence, 
            description="query_seq", 
            num_res=num_res
        )
        
        msa_features = pipeline.make_msa_features(
            msas=[[sequence]], 
            deletion_matrices=[[[0] * num_res]]
        )
        
        template_features = {
            'template_aatype': np.zeros((0, num_res), dtype=np.int32),
            'template_all_atom_positions': np.zeros((0, num_res, 37, 3), dtype=np.float32),
            'template_all_atom_masks': np.zeros((0, num_res, 37), dtype=np.float32),
            'template_domain_names': np.zeros((0,), dtype=object),
            'template_sequence': np.zeros((0,), dtype=object),
            'template_sum_probs': np.zeros((0,), dtype=np.float32),
        }
        
        feature_dict = {**sequence_features, **msa_features, **template_features}
        return feature_dict

    @torch.no_grad()
    def process_batch(self, sequences: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(sequences)
        max_len = max(len(seq) for seq in sequences)
        
        coordinates = torch.zeros((batch_size, max_len, 4, 3), dtype=torch.float32, device=self.device)
        mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=self.device)
        
        for batch_idx, seq in enumerate(sequences):
            seq_len = len(seq)
            
            mask[batch_idx, :seq_len] = True
            
            raw_features = self._generate_single_sequence_features(seq)
            
            processed_features = self.runner.process_features(raw_features, random_seed=42)
            
            prediction_outputs = self.runner.predict(processed_features)
            
            all_atom_positions = prediction_outputs['structure_module']['final_atom_positions']
            
            backbone_positions = all_atom_positions[:, self.target_atom_indices, :]  # Shape: (R, 4, 3)
            
            backbone_tensor = torch.from_numpy(backbone_positions).to(self.device)
            
            coordinates[batch_idx, :seq_len, :, :] = backbone_tensor
            
        return coordinates, mask


def evaluate_batch_rmsd(
    sequences: List[str], 
    ground_truth_coords: torch.Tensor, 
    gt_mask: torch.Tensor,
    colabfold_engine: 'ColabFoldValidationEngine'
) -> Dict[str, torch.Tensor]:

    pred_coords, pred_mask = colabfold_engine.process_batch(sequences)
    
    device = pred_coords.device
    B, R_pred, A, _ = pred_coords.shape
    _, R_gt, _, _ = ground_truth_coords.shape
    R_target = max(R_pred, R_gt)
    
    ground_truth_coords = ground_truth_coords.to(device)
    gt_mask = gt_mask.to(device)
    
    if R_pred < R_target:
        pad_size = R_target - R_pred
        pred_coords = torch.cat([pred_coords, torch.zeros((B, pad_size, A, 3), device=device)], dim=1)
        pred_mask = torch.cat([pred_mask, torch.zeros((B, pad_size), dtype=torch.bool, device=device)], dim=1)
        
    if R_gt < R_target:
        pad_size = R_target - R_gt
        ground_truth_coords = torch.cat([ground_truth_coords, torch.zeros((B, pad_size, A, 3), device=device)], dim=1)
        gt_mask = torch.cat([gt_mask, torch.zeros((B, pad_size), dtype=torch.bool, device=device)], dim=1)
        
    combined_mask = pred_mask & gt_mask  
    
    atom_mapping = {'C': 0, 'CA': 1, 'N': 2, 'O': 3}
    scores = {}
    
    for atom_name, atom_idx in atom_mapping.items():
        P_atom = pred_coords[:, :, atom_idx, :]
        Q_atom = ground_truth_coords[:, :, atom_idx, :]
        
        scores[f"{atom_name}_rmsd"] = kabsch_rmsd(P_atom, Q_atom, mask=combined_mask)
        
    P_backbone = pred_coords.reshape(B, -1, 3)
    Q_backbone = ground_truth_coords.reshape(B, -1, 3)
    
    backbone_mask = combined_mask.unsqueeze(-1).expand(-1, -1, 4).reshape(B, -1)
    
    scores["all_backbone_rmsd"] = kabsch_rmsd(P_backbone, Q_backbone, mask=backbone_mask)
    
    return scores


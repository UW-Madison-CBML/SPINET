import os
import torch
import numpy as np
from typing import List, Tuple, Dict
from alphafold.common import residue_constants
from alphafold.data import pipeline
from alphafold.model import config, model

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
    if mask:
        mask = mask.to(device)
     
    

     
    return 
 

 
   

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
):
    """

    params:
        sequences: lists of strings of seq1 amino acids
        ground_truth_coords: ground_truth time-series, shape (traj, n_frames, n_residues, n_atoms,3) # n_residues is padded and batched
        gt_mask: Mask from groun-truth.
    return:
        list of per-protein alignment scores (kabsch align at each timestep and then average RMSD over time)"""

    # alphafold prediction
    pred_coords, pred_mask = colabfold_engine.process_batch(sequences)
    
    device = pred_coords.device
    B, R_pred, A, _ = pred_coords.shape
    _, R_gt, _, _ = ground_truth_coords.shape
    R_target = max(R_pred, R_gt)
    
    ground_truth_coords = ground_truth_coords.to(device)
    gt_mask = gt_mask.to(device)
    
    # determine padding
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
    
        
    P_backbone = pred_coords.reshape(B, -1, 3)
    Q_backbone = ground_truth_coords.reshape(B, -1, 3)
    
    backbone_mask = combined_mask.unsqueeze(-1).expand(-1, -1, 4).reshape(B, -1)
    
    scores["all_backbone_rmsd"] = kabsch_rmsd(P_backbone, Q_backbone, mask=backbone_mask)
    
    return scores


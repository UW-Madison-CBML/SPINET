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
    assert Q.shape == P.shape, "P and Q do not have same shape"
    n = P.shape[-1]
    if mask:
        assert mask.any(dim=-1).all().item(), "each molecule must have at least one atom"

        mask = mask.to(device)

        mask = mask.unsqueeze(-1) # so that it broadcasts

        # center it 
        # this will not fail because we check that each molecule has at least one atom, thus mask.sum(dim=-1) >= 1
        P = P - ((P * mask).sum(dim=-2, keepdim=True) / mask.sum(-2, keepdim=True))
        Q = Q - ((Q * mask).sum(dim=-2, keepdim=True) / mask.sum(-2, keepdim=True))

        square_mask = mask * mask.mT # ... num_atoms, num_atoms

        # calculate covariance
        H = torch.einsum("...is,...js,...ij->...ij", P, Q, square_mask)
        
        # SVD
        U, _, Vt = torch.linalg.svd(H)

        # determinant
        d = torch.linalg.det(U) * torch.linalg.det(Vt)

        # this puts 1, ,...,  1, d on the diagonal
        d_diag = torch.diag_embed(torch.cat([torch.ones(*d.shape, n-1), d.unsqueeze(-1)], dim=-1))
        
        # calculate R
        R = torch.matmul(torch.matmul(U, d_diag), Vt)
       
        Q_rotated = torch.matmul(Q, R.mT)

        # calculate RMSD
        errors = torch.linalg.norm(P - Q_rotated, dim=-1)
        squared_errors = errors ** 2
        mask = mask.squeeze(0)
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
        d_diag = torch.diag_embed(torch.cat([torch.ones(*d.shape, n-1), d.unsqueeze(-1)], dim=-1)) # 3,3
        
        # calculate R
        R = torch.matmul(torch.matmul(U, d_diag), Vt)
       
        Q_rotated = torch.matmul(Q, R.mT)

        # calculate RMSD
        errors = torch.linalg.norm(P - Q_rotated, dim=-1)
        squared_errors = errors ** 2
        mean_squared_errors = squared_errors.mean(dim=-1)
        rmsd = torch.sqrt(mean_squared_errors) 
     
    return rmsd.cpu()
 

 
   

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

    pred_coords, pred_mask = colabfold_engine.process_batch(sequences)
    
    device = pred_coords.device
    B,    R, A, _ = pred_coords.shape
    _, T, _, _, _ = ground_truth_coords.shape
    
    ground_truth_coords = ground_truth_coords.to(device)
    gt_mask = gt_mask.to(device)
    
    P_backbone = pred_coords.reshape(B, R*A, 3)[:, None, :, :].expand(-1, T, -1, -1)
    Q_backbone = ground_truth_coords.reshape(B, T, R*A, 3) 
   
    backbone_mask = gt_mask[:,:,:,None].repeat(-1, -1, -1, A).reshape(B, T, R*A)
    
    return kabsch_rmsd(P_backbone, Q_backbone, mask=backbone_mask)


import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import torch.nn.functional as F
import pandas as pd
from Bio.Data import IUPACData
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse
import numpy as np
from typing import Union, Tuple
from load_dynamics import FEATURE_COLUMNS as features
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os

# this is a combination k-NN and distance threshold, generalized to use arbitrary dist mats
def edge_index_from_distmat(dist_matrix: torch.Tensor, epsilon: float, k:int=32):
    assert dist_matrix.shape[0] == dist_matrix.shape[1], f"dist_matrix is not square: {dist_matrix.shape}"
    N = dist_matrix.shape[0]
    # change k so topk doesn't fail
    k = min(k, N)
    masked = dist_matrix.clone()
    masked.fill_diagonal_(float('inf'))
    masked[masked > epsilon] = float('inf')
    topk_dist, topk_idx = torch.topk(masked, k=k, largest=False, dim=-1)
    valid = topk_dist.isfinite()
    row = torch.arange(N)[:,None].expand(-1, k)[valid]
    col = topk_idx[valid]
    edge_weight = topk_dist[valid]
    adj = torch.zeros(N, N)
    adj[row, col] = edge_weight
    adj = torch.maximum(adj, adj.t())
    edge_index, _ = dense_to_sparse(adj)
    return edge_index


def calculate_node_features(coords, feats, labels, mask, atom_indices, frame_origin, not_frame_origin_mask, epsilon=5.0, add_temporal_edges=True):

    pos = torch.as_tensor(coords, dtype=torch.float32)
    x = torch.as_tensor(feats, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.long)
    node_mask = torch.as_tensor(mask, dtype=torch.bool)


    carbon_alphas = x[:,:,atom_indices[frame_origin]]

    u,v = carbon_alphas - x[:,:,atom_indices["C"]], x[:,:,atom_indices["N"]] - carbon_alphas

    x_basis = u - v
    # normalize
    x_basis = x_basis / torch.clamp(torch.norm(x_basis, dim = -1, keepdim=True), min=0.01)

    y_basis = torch.cross(u, v, dim = -1) 
    # normalize
    y_basis = y_basis / torch.clamp(torch.norm(y_basis, dim=-1, keepdim=True), min=0.01)
    
    # get final orthonormal basis vector:
    z = torch.cross(x_basis,y_basis, dim=-1) # will be normal since other two vectors are normal 
    
    raw_to_basis_matrix = torch.stack([x_basis,y_basis,z], dim=-2) # num_res, n_frames, 3, 3

    # positions will contain the atomic coordinate

    pos_feats = x[:, :, :3*len(atom_indices)]          # num_res, n_frames, 3*num_atoms
    positions = pos_feats[:,:,:len(atom_indices)*3].view(x.shape[0], x.shape[1],len(atom_indices), 3)

    backbone_centroids = positions.mean(dim=2)
    unpadded = backbone_centroids.diff(dim=1)
    velocities = torch.cat([unpadded[:,:1,:], unpadded],dim=1)


    # features will be certain positions in the coordinate frame
    relative_features = positions[:, :, not_frame_origin_mask, :] - positions[:, :, ~not_frame_origin_mask, :] # num_res, n_frames, num_atoms-1, 3
    in_frame_features = torch.matmul(relative_features, raw_to_basis_matrix).view(x.shape[0], x.shape[1], 3*(len(atom_indices)-1)) # num_res, n_frames, (num_atoms-1) * 3
    in_frame_velocities = torch.matmul(raw_to_basis_matrix.mT, velocities.unsqueeze(-1)).squeeze(-1)
    
    features = torch.cat([in_frame_features, x[:,:,len(atom_indices):], in_frame_velocities], dim=-1)

    return {"x":x, "features":features, "velocities":velocities, "frame_mats":raw_to_basis_matrix, "pos":pos, "y":y, "node_mask":node_mask}
    

def get_node_features(traj, pos_cols, feature_cols, amino_acids, epsilon, atom_indices, frame_origin, not_frame_origin_mask):

    frames = [group for name,group in list(traj)]
    
    pos = np.stack([frame[pos_cols].to_numpy() for frame in frames], axis=1)


    return calculate_node_features(
            pos,
            np.stack([frame[feature_cols].to_numpy() for frame in frames], axis=1),
            torch.tensor([amino_acids.index(res) for res in frames[0]['residue'].to_list()], dtype=torch.long),
            frames[0]["mask"].to_numpy(),
            atom_indices, 
            frame_origin, 
            not_frame_origin_mask,
            epsilon = epsilon
    )



   

    
class ResidueClassifierDataset(Dataset):

    # ground truth order of amino acid indices. they must be capitalized
    AMINO_ACIDS = [code.upper() for code in IUPACData.protein_letters_3to1.keys()]
    FEATURE_COLS = features
    POS_COLS = ["CA_x","CA_y","CA_z"]
 
    
    #--------------------------------------------------
    # df should be loaded in with the pdb_id col added, and then validation set formed by splitting out along that column. Want to make a protein in the validation set has never been seen before
    # TODO plot histogram of epsilon
    
    def __init__(self, df, atoms, atom_indices, frame_origin="CA", traj_len:None|int=200, variable_length:None|tuple[int,int]=None, epsilon:float=5.0, fixed_length:None|int=None):
        """
        self
        df: the dataframe containing trajectory information 
        traj_len: the fixed length of all trajectories if it exists
        variable_length: None if fixed length sequences else the range (inclusive) of valid sequence sizes
        epsilon: tolerance to build edge between nodes, i.e. if during the trajectory the edges ever get within epsilon from eachother
        """
        self.df = df
        self.groups = [group.groupby('timestep') for _, group in list(df.groupby("traj_id"))]
        self.epsilon = epsilon
        assert (variable_length is None) == (fixed_length is not None), "the following does not hold: variable_length is None XOR fixed_length is None"
        self.fixed_length = fixed_length
        self.variable_length = variable_length
        self.traj_len = traj_len
        self.atoms = atoms
        self.atom_indices = atom_indices
        assert set(self.atoms) == set(self.atom_indices.keys()), "atoms are not the keys of atom_indices"
        assert frame_origin in self.atoms, f"frame origin: {frame_origin} is not in atoms"
        self.frame_origin = frame_origin
        self.not_frame_origin_mask = torch.tensor([atom != self.frame_origin for atom in self.atoms], dtype=torch.bool)
 

        if self.traj_len is not None and self.fixed_length is not None:
            self.index = np.stack(np.broadcast_arrays(np.arange(len(self.groups))[:,None],np.arange(self.traj_len - (self.fixed_length - 1))[None,:], self.fixed_length + np.arange(self.traj_len - (self.fixed_length - 1))[None,:]), axis=-1).reshape(-1, 3)
        elif self.fixed_length is not None: 
            self.index = []
            for i, traj in enumerate(self.groups):
                length = len(traj) 
                self.index.append(np.stack(np.broadcast_arrays(i,np.arange(length - (self.fixed_length - 1)), self.fixed_length + np.arange(length - (self.fixed_length - 1))), axis=-1))
            self.index = np.concatenate(self.index, axis=0)
        else:
            self.index = []
            min_len, max_len = self.variable_length
            for i, traj in enumerate(self.groups):
                length = len(traj) 
                for seq_len in range(self.variable_length[0], self.variable_length[1] + 1): # upper bound on lengths is inclusive
                    for j in range(length-(seq_len - 1)):
                        self.index.append((i,j,j+seq_len))
            self.index = np.array(self.index)
        self.trajs = []
        max_workers = os.cpu_count()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:

            futures = [executor.submit(get_node_features, traj, self.__class__.POS_COLS, self.__class__.FEATURE_COLS, self.__class__.AMINO_ACIDS, self.epsilon, self.atom_indices, self.frame_origin, self.not_frame_origin_mask) for traj in self.groups]
            
            kwargs = {"total": len(futures), "desc": "Processing Training Data Seqs", "unit": "job"}
            for future in tqdm(as_completed(futures), **kwargs):
                try:
                    data = future.result()
                    self.trajs.append(data)
                except Exception as e:
                    print(f"Worker generated an exception: {e}")
     
            

    def __len__(self):
        return len(self.index) 

    def __getitem__(self, idx):
        traj_idx, frame_index_start, frame_index_end = self.index[idx]
        idxs = slice(frame_index_start, frame_index_end)
        traj = self.trajs[traj_idx]
        length = torch.as_tensor(frame_index_end-frame_index_start, dtype=torch.long)
        
        x, features, pos, velocities, y, node_mask = traj["x"][:,idxs], traj["features"][:,idxs], traj["pos"][:,idxs], traj["velocities"][:,idxs], traj["y"],traj["node_mask"]
        pos_time_first = pos.permute(1,0,2).contiguous()
        dists_over_time = torch.cdist(pos_time_first, pos_time_first, p=2.0)

        dists = dists_over_time.amin(dim=0)

        edge_index = edge_index_from_distmat(dists, epsilon=self.epsilon, k=32)
        edge_attr = torch.zeros(edge_index.shape[1], 1)

        if pos.shape[0] > 1: # check that protein isn't a monomer
            idx = torch.arange(pos.shape[0] - 1)
            temporal = torch.stack([
                torch.cat([idx, idx + 1]),
                torch.cat([idx + 1, idx]),
            ])
            temporal_attr = torch.ones(temporal.shape[1], 1)
            edge_index = torch.cat([edge_index, temporal], dim=1)
            edge_attr = torch.cat([edge_attr, temporal_attr], dim=0)


        # now let's build edge features given by pairwise distances between atoms
        pos_feats = x[:, :, :3*len(self.atom_indices)]          # num_res, n_frames, 3*num_atoms
        edges = torch.stack([pos_feats[edge_index[0]], pos_feats[edge_index[1]]], dim=1)
        edges = edges.view(edge_index.shape[1], 2, x.shape[1], len(self.atom_indices), 3)
        dist_features = torch.cdist(edges[:,0], edges[:,1]) # E, n_frames, len(atom_indices), len(atom_indices)
         

        pair_wise_velocities = F.cosine_similarity(velocities[edge_index[0]], velocities[edge_index[1]], dim=-1).unsqueeze(-1)
        edge_features = torch.cat([dist_features.view(edge_index.shape[1],x.shape[1], len(self.atom_indices)**2), edge_attr[:,None,:].expand(-1,x.shape[1], -1), pair_wise_velocities], dim=2)

        return Data(x=features, y=y, pos=pos,edge_index=edge_index, edge_attr =edge_features, node_mask=node_mask, lengths=length.unsqueeze(0)) # need to add single dim so that concat works properly

        


    def graph_collate(self, batch):
        if self.fixed_length is not None:
            out_data = Batch.from_data_list(batch)
            return out_data.sort()
        else:
            lengths = torch.cat([data.lengths for data in batch])
            max_len = lengths.amax().item()  
             
            for i, length in enumerate(lengths):
                pad_amt = max_len - length
                
                pad_config = (0, 0,  0, pad_amt,  0, 0) 
                
                batch[i].x = F.pad(batch[i].x, pad_config, value=0.0)
                batch[i].pos = F.pad(batch[i].pos, pad_config, value=0.0)

            out_data = Batch.from_data_list(batch)
            out_data.lengths = lengths

            return out_data.sort()

    @staticmethod   
    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2**32

        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset

        dataset.rng = np.random.default_rng(seed=worker_seed)            
            

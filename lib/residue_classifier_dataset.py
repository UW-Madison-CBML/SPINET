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


def calculate_node_features(coords, feats, labels, mask, name, epsilon=5.0):

    pos = torch.as_tensor(coords, dtype=torch.float32)
    x = torch.as_tensor(feats, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.long)
    node_mask = torch.as_tensor(mask, dtype=torch.bool)


    return {"x":x, "pos":pos, "y":y, "node_mask":node_mask, "traj_id":name}
    

def get_node_features(traj, pos_cols, feature_cols, amino_acids, epsilon, name):

    frames = [group for name,group in list(traj)]
    
    pos = np.stack([frame[pos_cols].to_numpy() for frame in frames], axis=1)


    return calculate_node_features(
            pos,
            np.stack([frame[feature_cols].to_numpy() for frame in frames], axis=1),
            torch.tensor([amino_acids.index(res) for res in frames[0]['residue'].to_list()], dtype=torch.long),
            frames[0]["mask"].to_numpy(),
            name,
            epsilon = epsilon
    )



   

    
class ResidueClassifierDataset(Dataset):

    # ground truth order of amino acid indices. they must be capitalized
    AMINO_ACIDS = [code.upper() for code in IUPACData.protein_letters_3to1.keys()]
    FEATURE_COLS = ["phi","psi","omega"]
    POS_COLS = ["CA_x","CA_y","CA_z"]
 
    
    #--------------------------------------------------
    # df should be loaded in with the pdb_id col added, and then validation set formed by splitting out along that column. Want to make a protein in the validation set has never been seen before
    # TODO plot histogram of epsilon
    
    def __init__(self, df, traj_len:None|int=200, variable_length:None|tuple[int,int]=None, epsilon:float=5.0, fixed_length:None|int=None):
        """
        self
        df: the dataframe containing trajectory information 
        traj_len: the fixed length of all trajectories if it exists
        variable_length: None if fixed length sequences else the range (inclusive) of valid sequence sizes
        epsilon: tolerance to build edge between nodes, i.e. if during the trajectory the edges ever get within epsilon from eachother
        """
        self.df = df
        self.groups = [(name, group.groupby('timestep')) for name, group in list(df.groupby("traj_id"))]
        self.epsilon = epsilon
        assert (variable_length is None) == (fixed_length is not None), "the following does not hold: variable_length is None XOR fixed_length is None"
        self.fixed_length = fixed_length
        self.variable_length = variable_length
        self.traj_len = traj_len
 

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
            futures = [executor.submit(get_node_features, traj, self.__class__.POS_COLS, self.__class__.FEATURE_COLS, self.__class__.AMINO_ACIDS, self.epsilon, name) for name, traj in self.groups]
            
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
        
        x, pos, y, node_mask, traj_id = traj["x"][:,idxs], traj["pos"][:,idxs], traj["y"], traj["node_mask"], traj["traj_id"]

        x_time_first = x.permute(1,0,2).contiguous()
        
        mean = x_time_first.mean(dim=0)
        std_dev = x_time_first.std(dim=0)
        deviations = x_time_first - mean
        skewness = torch.mean(deviations ** 3, dim=0) / (torch.clamp(std_dev, min=0.001) ** 3)
        kurtosis = torch.mean(deviations ** 4, dim=0) / (torch.clamp(std_dev, min=0.001) ** 4) 
        x = torch.cat([mean,std_dev,skewness,kurtosis], dim=-1)
        
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

        return Data(x=x, y=y, pos=pos, edge_index=edge_index, node_mask=node_mask, traj_id=traj_id, index=torch.tensor([[traj_idx, frame_index_start, frame_index_end]]))

        


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
            

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

def build_graph(coords, feats, labels, mask, length, epsilon=5.0, add_temporal_edges=True):

    pos = torch.as_tensor(coords, dtype=torch.float32)
    x = torch.as_tensor(feats, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.long)
    node_mask = torch.as_tensor(mask, dtype=torch.bool)
    length = torch.as_tensor(length, dtype=torch.long)

    dists = torch.cdist(pos, pos, p=2.0).amin(dim=0) # min so that any two nodes that are ever connected will have an direct edge

    edge_index = edge_index_from_distmat(dists, epsilon=epsilon, k=32)
    edge_attr = torch.zeros(edge_index.shape[1], 1)

    if pos.size(0) > 1: # check that protein isn't a monomer
        idx = torch.arange(pos.size(0) - 1)
        temporal = torch.stack([
            torch.cat([idx, idx + 1]),
            torch.cat([idx + 1, idx]),
        ])
        temporal_attr = torch.ones(temporal.shape[1], 1)
        edge_index = torch.cat([edge_index, temporal], dim=1)
        edge_attr = torch.cat([edge_attr, temporal_attr], dim=0)

    return Data(x=x, pos=pos, y=y, node_mask=node_mask, edge_index=edge_index, edge_attr=edge_attr, lengths=length.unsqueeze(0))



class ResidueClassifierDataset(Dataset):

    # ground truth order of amino acid indices. they must be capitalized
    AMINO_ACIDS = [code.upper() for code in IUPACData.protein_letters_3to1.keys()] + ["PYL", "SEC"] # add pyrrolysine and selenocysteine
    FEATURE_COLS = ["dx", "dy", "dz", "bond_ang", "bond_len"]
    POS_COLS = ["x","y","z"]
    

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
        self.groups = [group.groupby('timestep') for _, group in list(df.groupby("traj_id"))]
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
        



    def __len__(self):
        return len(self.index) 

    def __getitem__(self, idx):
        group_idx, frame_idx_start, frame_idx_end = self.index[idx]
        traj = self.groups[group_idx]
        frame_idxs = range(frame_idx_start, frame_idx_end) # not inclusive

        if(self.traj_len is not None):
            assert len(traj) == self.traj_len, "TRAJ LEN does not match length of trajectory"

        frames = [traj.get_group(i) for i in frame_idxs]
        
        pos = np.stack([frame[self.__class__.POS_COLS].to_numpy() for frame in frames], axis=1)
        pos -= pos.mean(axis=(0,1)) # avg COM over time
        # I'm just using the first frame's mask as the mask, that way GT mask in the df remains

        length = len(frames)
        return build_graph(
                pos,
                np.stack([frame[self.__class__.FEATURE_COLS].to_numpy() for frame in frames], axis=1),
                torch.tensor([self.__class__.AMINO_ACIDS.index(res) for res in frames[0]['residue'].to_list()], dtype=torch.long),
                frames[0]["mask"].to_numpy(),
                length,
                epsilon = self.epsilon
        )


    def graph_collate(self, batch):
        if self.fixed_length is not None:
            return Batch.from_data_list(batch)
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
            return out_data

             
            
            

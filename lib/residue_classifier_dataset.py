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
from load_dynamics import BACKBONE_ATOMS
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import h5py

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

    
class ResidueClassifierDataset(Dataset):

    # ground truth order of amino acid indices. they must be capitalized
    AMINO_ACIDS = [code.upper() for code in IUPACData.protein_letters_3to1.keys()]
    FRAME_ORIGIN = "CA"
    ATOMS = BACKBONE_ATOMS
    ATOM_INDICES = {atom: i for i, atom in enumerate(ATOMS)}
    REQUIRED_DATASETS = {"coordinates", "dihedrals", "spinet_features","frame_maps","residues"}
    #--------------------------------------------------
    # df should be loaded in with the pdb_id col added, and then validation set formed by splitting out along that column. Want to make a protein in the validation set has never been seen before
    # TODO plot histogram of epsilon
    
    def __init__(self, h5_path, np_rng, groups:list|None=None, paradigm:str="dynamic", traj_len:None|int=200, variable_length:None|tuple[int,int]=None, epsilon:float=5.0, fixed_length:None|int=None):
        """
        self
        h5_path: the dataframe containing trajectory information 
        traj_len: the fixed length of all trajectories if it exists
        variable_length: None if fixed length sequences else the range (inclusive) of valid sequence sizes
        epsilon: tolerance to build edge between nodes, i.e. if during the trajectory the edges ever get within epsilon from eachother
        """
        self.h5_path = h5_path
        self.h5_file = None 
        self.groups = groups
        self.epsilon = epsilon
        self.np_rng = np_rng
        assert paradigm in ["dynamic", "static", "ensemble"], f"invalid option for paradigm: {paradigm}"
        

        # TODO implement and get rid of below
        assert paradigm != "ensemble", "ensemble is not implemented"
        self.paradigm = paradigm
        if self.paradigm == "dynamic":
            assert (variable_length is None) == (fixed_length is not None), "the following does not hold: variable_length is None XOR fixed_length is None"
        elif self.paradigm == "static":
            assert fixed_length == 1, "fixed length must be 1 if using static model"

        self.fixed_length = fixed_length
        self.variable_length = variable_length
        self.traj_len = traj_len


        with h5py.File(self.h5_path, "r") as f:

            self.build_groups(f)
            if self.paradigm == "dynamic":
                self.index = self.build_dynamic_index(f)
            elif self.paradigm == "static":
                self.index = self.build_static_index(f)
            # else: otherwise the program will fail. TODO implement this
        
     
            

    def __len__(self):
        return len(self.index) 

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r", libver="latest", swmr=True)
                
        if self.paradigm == "dynamic":
            traj_idx, frame_index_start, frame_index_end = self.index[idx]
            idxs = slice(frame_index_start, frame_index_end)
        elif self.paradigm == "static":
            traj_idx, idxs = self.index[idx]
        # else: TODO 

        group_name = self.groups[traj_idx]
        
        # ensure that these have the proper shape for the paradigm
        if self.paradigm == "ensemble":
            pass 
        else:
            coordinates = torch.from_numpy(self.h5_file[group_name + "/coordinates"][:, idxs])
            pos = coordinates[:,:, self.__class__.ATOM_INDICES[self.__class__.FRAME_ORIGIN]] if self.paradigm == "dynamic" else coordinates[:, self.__class__.ATOM_INDICES[self.__class__.FRAME_ORIGIN]]
            features = torch.from_numpy(self.h5_file[group_name + "/spinet_features"][:, idxs])
            frame_maps = torch.from_numpy(self.h5_file[group_name + "/frame_maps"][:, idxs])
            y = torch.tensor([self.__class__.AMINO_ACIDS.index(res.decode()[:3]) for res in self.h5_file[group_name + "/residues"][:]])
            node_mask = torch.zeros(len(y))
            traj_id = group_name

        if self.paradigm == "dynamic":
            pos_time_first = pos.permute(1,0,2)
            dists_over_time = torch.cdist(pos_time_first, pos_time_first, p=2.0)

            dists = dists_over_time.amin(dim=0)

        elif self.paradigm == "static":
            dists = torch.cdist(pos, pos, p=2.0)

        edge_index = edge_index_from_distmat(dists, epsilon=self.epsilon, k=32)
        edge_attr = (torch.abs(edge_index[0] - edge_index[1]) == 1)[:,None].float() # this way we don't have double edges. Not that double edges are necessarily bad but imposing this restriction helps sheaf Laplacian be more well-behaved

        return Data(x=features, y=y, pos=coordinates, frame_maps = frame_maps, edge_index=edge_index, edge_attr=edge_attr, node_mask=node_mask, traj_id=traj_id, index=torch.tensor([self.index[idx]]))

    def build_groups(self, h5_file):
        use_split = self.groups is not None
        split = self.groups 
        self.groups = []
        def visit(name, obj):
            if isinstance(obj, h5py.Group) and all(ds in obj for ds in self.__class__.REQUIRED_DATASETS) and (use_split <= (name.split("/")[0] in split)):
                self.groups.append(name) 
        h5_file.visititems(visit)


    def build_dynamic_index(self, h5_file):
        index = []
        if self.traj_len is not None and self.fixed_length is not None:
            index = np.stack(np.broadcast_arrays(np.arange(len(self.groups))[:,None],np.arange(self.traj_len - (self.fixed_length - 1))[None,:], self.fixed_length + np.arange(self.traj_len - (self.fixed_length - 1))[None,:]), axis=-1).reshape(-1, 3)
        elif self.fixed_length is not None: 
            index = []
            for i, group_name in enumerate(self.groups):
                length = 200 # h5_file[group_name + "/" + "coordinates"].shape[1]
                index.append(np.stack(np.broadcast_arrays(i,np.arange(length - (self.fixed_length - 1)), self.fixed_length + np.arange(length - (self.fixed_length - 1))), axis=-1))
            index = np.concatenate(index, axis=0)
        else:
            index = []
            min_len, max_len = self.variable_length
            for i, group_name in enumerate(self.groups):
                length = 200 # h5_file[group_name + "/" + "coordinates"].shape[1]
                for seq_len in range(self.variable_length[0], self.variable_length[1] + 1): # upper bound on lengths is inclusive
                    for j in range(length-(seq_len - 1)):
                        index.append((i,j,j+seq_len))
            index = np.array(index)
        return index

    def build_static_index(self, h5_file):
        index = []
        for i, group_name in enumerate(self.groups):
            j = int(self.np_rng.random() * h5_file[group_name + "/" + "coordinates"].shape[1]) # pick random timepoints
            index.append((i, j))
        return index
        


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
            

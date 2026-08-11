import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import pandas as pd
from Bio.Data import IUPACData
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse

# this is a combination k-NN and distance threshold, generalized to use arbitrary dist mats
def edge_index_from_distmat(dist_matrix: torch.Tensor, epsilon: float, k:int=32):
    N = dist_matrix.size(0)
    masked = dist_matrix.clone()
    masked.fill_diagonal_(float('inf')) 
    masked[masked > epsilon] = float('inf')
    topk_dist, topk_idx = torch.topk(masked, k=k, largest=False, dim=-1)
    valid = topk_dist.isfinite()
    row = torch.arange(N).unsqueeze(1).expand(-1, k)[valid]
    col = topk_idx[valid]
    edge_weight = topk_dist[valid]
    adj = torch.zeros(N, N)
    adj[row, col] = edge_weight
    adj = torch.maximum(adj, adj.t())
    edge_index, _ = dense_to_sparse(adj)
    return edge_index

def build_graph(coords, feats, labels, mask, epsilon=5.0, add_temporal_edges=True):
 
    pos = torch.as_tensor(coords, dtype=torch.float32)
    x = torch.as_tensor(feats, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.long)
    node_mask = torch.as_tensor(mask, dtype=torch.bool)

    dists = torch.cdist(pos, pos, p=2.0).min(dim=0) # min so that any two nodes that are ever connected will have an direct edge
 
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
 
    return Data(x=x, pos=pos, y=y, node_mask=node_mask, edge_index=edge_index, edge_attr=edge_attr)



class ResidueClassifierDataset(Dataset):
   
    # ground truth order of amino acid indices. they must be capitalized
    AMINO_ACIDS = [code.upper() for code in IUPACData.protein_letters_3to1.keys()] + ["PYL", "SEC"] # add pyrrolysine and selenocysteine
    FEATURE_COLS = ["dx", "dy", "dz", "bond_ang", "bond_len"]
    POS_COLS = ["x","y","z"]
    # CAUTION this is hardcoded 
    TRAJ_LEN = 200


    #-------------------------------------------------- 
    # df should be loaded in with the pdb_id col added, and then validation set formed by splitting out along that column. Want to make a protein in the validation set has never been seen before
    # TODO plot histogram of epsilon
    def __init__(self, df, np_rng, epsilon=5.0, timesteps=16):
        self.df = df
        self.groups = [group.groupby('timestep') for _, group in list(df.groupby("traj_id"))]
        self.epsilon = epsilon 
        self.timesteps = timesteps
        self.np_rng = np_rng

    def __len__(self):
        return (self.__class__.TRAJ_LEN - (self.timesteps - 1)) * len(self.groups)

    def __getitem__(self, idx):
        traj = self.groups[idx // (self.__class__.TRAJ_LEN - (self.timesteps - 1))]
        traj_window_idx = idx % (self.__class__.TRAJ_LEN - (self.timesteps - 1))
        assert len(trajs) == self.__class__.TRAJ_LEN, "TRAJ LEN does not match length of trajectory"
        frames = [traj.get(traj_window_idx + i) for i in range(timesteps)]
        pos = np.stack([frame[self.__class__.POS_COLS].to_numpy() for frame in frames], axis=1)
        pos -= pos.mean(axis=(0,1)) # avg COM over time
        # I'm just using the first frame's mask as the mask, that way GT mask in the df remains
        return build_graph(pos, np.stack([frame[self.__class__.FEATURE_COLS].to_numpy() for frame in frames], axis=1), torch.tensor([self.__class__.AMINO_ACIDS.index(res) for res in frames[0]['residue'].to_list()], dtype=torch.long), frames[0]["mask"].to_numpy(), self.epsilon)
 
 
    @staticmethod
    def graph_collate(batch):
        return Batch.from_data_list(batch)

        

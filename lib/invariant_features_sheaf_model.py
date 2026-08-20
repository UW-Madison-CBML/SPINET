import torch
import torch_sparse
from torch.autograd.gradcheck import gradcheck
from typing import Tuple
from torch_geometric.nn.conv import GATConv
from torch_geometric.nn import MessagePassing
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax, sort_edge_index
from torch.func import functional_call, vmap
from torch.nn.utils.parametrizations import orthogonal

#-------------------------------------------------------
# sheaf learners

class SheafLearnerLowRank(nn.Module):
    def __init__(self, input_dim: int, stalk_dim: int, rank: int):
        super().__init__()
        assert rank <= stalk_dim, "Rank 'r' cannot be greater than stalk dimension 'd'."
        self.input_dim = input_dim
        self.stalk_dim = stalk_dim
        self.rank = rank

        self.lin = nn.Linear(self.input_dim * 2,self.input_dim * 2)
        self.map_learner = nn.Linear(self.input_dim * 2, 2 * self.stalk_dim * self.rank)

    def forward(self, x, edge_index):
        row, col = edge_index
        edge_features = torch.cat([x[row], x[col]], dim=-1)
        maps = self.map_learner(F.relu(self.lin(edge_features)))
        left_maps = maps[:, :self.stalk_dim*self.rank].view(edge_index.shape[1], self.stalk_dim, self.rank)
        right_maps = maps[:, self.stalk_dim*self.rank:].view(edge_index.shape[1], self.rank, self.stalk_dim)
        return torch.matmul(left_maps, right_maps)
         

class SheafLearnerOrthogonal(nn.Module):
    def __init__(self, input_dim: int, stalk_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.stalk_dim = stalk_dim

        self.lin = nn.Linear(self.input_dim * 2, self.input_dim)
        self.tri_dim = ((self.stalk_dim-1) * (self.stalk_dim) )// 2
        self.triu_learner = nn.Linear(self.input_dim, self.tri_dim)

        # now we need to build a matrix that will enter in each learned value and then unflatten the matrix
        indices = torch.triu_indices(self.stalk_dim, self.stalk_dim, offset=1)
        indices = indices[0] + (self.stalk_dim * indices[1])
        assert self.tri_dim == indices.shape[0], "indices shape isn't the same as triangle dim"
        upper_indices_matrix = torch.zeros(self.stalk_dim**2, self.tri_dim) 
        upper_indices_matrix[indices[:,None], torch.arange(self.tri_dim)] = 1
        self.register_buffer("upper_indices_matrix",upper_indices_matrix)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        row, col = edge_index
        edge_features = torch.cat([x[row], x[col]], dim=-1)
        upper_tri = self.triu_learner(F.relu(self.lin(edge_features)))

        # put upper triangle values in
        maps_flat = torch.matmul(self.upper_indices_matrix, upper_tri[:,None,:])

        # make then square
        maps = maps_flat.view(edge_index.shape[1], self.stalk_dim, self.stalk_dim)

        # make them skew symmetric
        maps = maps - maps.mT

        # we need to scale maps or else the Neumann series will blow up
        # \|M\|_2 (i.e. p=2 spectral norm) <= \|M\|_F
        fro_norm = torch.linalg.norm(maps, ord='fro', dim=(1,2), keepdim=True)
        scale = torch.clamp(fro_norm / 0.9, min=1.0)
        maps = maps / scale

        # output needs to be (I - M)(I + M)^-1
        I = torch.eye(self.stalk_dim, device=maps.device)

        # now we use a short Neumann series approximation (I + M = I - (-M))
        left_mat = I - maps
        maps_squared = torch.matmul(maps, maps)
        maps = torch.matmul(left_mat, left_mat + maps_squared - torch.matmul(maps, maps_squared))
        
        return maps

class SheafLearner(nn.Module):
    def __init__(self, input_dim:int, stalk_dim:int):
        super().__init__()
        self.input_dim = input_dim
        self.stalk_dim = stalk_dim
        self.lin = nn.Linear(2*self.input_dim, 2*self.stalk_dim) # TODO is this too much?
        self.map_learner = nn.Linear(2*self.stalk_dim, self.stalk_dim ** 2)

    def forward(self, x, edge_index):
        row, col = edge_index

        x_row = x[row]
        x_col = x[col]

        maps = self.map_learner(F.relu(self.lin(torch.cat([x_row, x_col], dim=-1)))).view(edge_index.shape[1], self.stalk_dim, self.stalk_dim)

        return maps


#-----------------------------------------------------------------------------------
class SheafAttentionConv(MessagePassing):
    def __init__(self, hidden_dim: int, stalk_dim: int, num_heads:int, dropout: float = 0.2, ablate_sheaves=False, restriction_map_type="low_rank"):
        super().__init__(aggr='add', node_dim=0)
        self.hidden_dim = hidden_dim
        self.stalk_dim = stalk_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_channels = self.hidden_dim // self.stalk_dim
        self.ablate_sheaves = ablate_sheaves
        self.restriction_map_type=restriction_map_type

        self.W_weights = nn.Parameter(torch.empty(self.num_heads, self.stalk_dim, self.stalk_dim))
        self.att_weights = nn.Parameter(torch.empty(self.num_heads, 1, 2 * self.hidden_dim))
        nn.init.xavier_uniform_(self.W_weights)
        nn.init.xavier_uniform_(self.att_weights)
        self._stateless_W = nn.Linear(self.stalk_dim, self.stalk_dim, bias=False)
        self._stateless_att = nn.Linear(2 * self.hidden_dim, 1, bias=False)

        self.apply_W = vmap(
            lambda params, tensor: functional_call(self._stateless_W, params, tensor),
            in_dims=({"weight": 0}, 0)
        )

        self.apply_att = vmap(
            lambda params, tensor: functional_call(self._stateless_att, params, tensor),
            in_dims=({"weight": 0}, 0)
        )

        self.project_concat = nn.Linear(self.num_heads * self.hidden_dim, self.hidden_dim)
        self.leaky = nn.LeakyReLU(0.2)
        if not self.ablate_sheaves:
            if self.restriction_map_type == "low_rank":
                self.sheaf_learner = SheafLearnerLowRank(self.hidden_dim, self.stalk_dim, self.stalk_dim//2)
            elif self.restriction_map_type == "orthogonal":
                self.sheaf_learner = SheafLearnerOrthogonal(self.hidden_dim, self.stalk_dim)
            else:
                self.sheaf_learner = SheafLearner(self.hidden_dim, self.stalk_dim)
                



    def forward(self, x, edge_index):
        if not self.ablate_sheaves: 
            node_to_edge_maps = self.sheaf_learner(x, edge_index)
            # we need to get a map from the index of edge (a,b) to the index of edge (b,a) to learn the transport maps F_{b \unlhd e_{a,b}}^T @ F_{a \unlhd e_{a,b}} 
            # edge_index comes in sorted so we sort again and keep track of the map by sorting an arange
            _, reverse_edge_indices = sort_edge_index(torch.roll(edge_index,1,0), torch.arange(edge_index.shape[1], device=x.device, dtype=torch.int64))
            edge_to_node_maps = node_to_edge_maps[reverse_edge_indices].mT # get transpose as to "invert" the map
            
            transport_maps = torch.matmul(edge_to_node_maps, node_to_edge_maps) # now this is the sheaf generalization of the adjacency map written A_\mathcal{F}
        else:
            # set transport maps to identity to ablate sheaves
            transport_maps = torch.eye(self.stalk_dim, device=x.device)[None, :, :].expand(edge_index.shape[1], -1, -1)

        x_stalk = x.view(x.shape[0], self.num_channels, self.stalk_dim) # this should automatically fail if the stalk_dim is input wrong
        
        out = self.propagate(edge_index, x=x, x_stalk=x_stalk, maps=transport_maps)
        return self.project_concat(out.view(x.shape[0], self.num_heads * self.hidden_dim))

    def message(self, x_i, x_j, x_stalk_j, maps, index, ptr, size_i):
        W_params = {"weight": self.W_weights}
        att_params = {"weight": self.att_weights}
        
        edge_features = torch.cat([x_i, x_j], dim=-1) 
        edge_features_batched = edge_features[None,None,:,:].expand(self.num_heads,-1,-1, -1) 
        
        x_stalk_j_batched = x_stalk_j[None,:,:,:].expand(self.num_heads,-1, -1, -1)

        alpha = self.leaky(self.apply_att(att_params, edge_features_batched))
        
        transformed = self.apply_W(W_params, x_stalk_j_batched)
        
        alpha = softmax(alpha, index, ptr, num_nodes=size_i, dim=2) 
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        if not self.ablate_sheaves: 
            maps_expanded = maps[None,:,:,:].expand(self.num_heads, -1, -1, -1)
            transported = torch.matmul(maps_expanded, transformed.mT)
        else: 
            transported = transformed
        alpha = alpha.permute(0,2,1,3).contiguous() # this will end up with num_heads, num_edges, 1, 1 which will broadcast
        return (alpha * transported).permute(1,0,2,3).contiguous()
    
class SheafResidualSANBlock(nn.Module):
    def __init__(self, hidden_dim: int, stalk_dim: int, num_heads: int, dropout: float = 0.2, ablate_sheaves=False, restriction_map_type="low_rank"):
        super().__init__()
        self.ablate_sheaves = ablate_sheaves
        self.hidden_dim = hidden_dim
        self.stalk_dim = stalk_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.restriction_map_type=restriction_map_type

        self.san = SheafAttentionConv(self.hidden_dim, self.stalk_dim, self.num_heads, dropout=self.dropout, ablate_sheaves=ablate_sheaves, restriction_map_type=self.restriction_map_type)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.act = nn.ReLU()

    def forward(self, data):
        residual = data.x
        out = self.san(data.x, data.edge_index)
        data.x = self.norm(residual + self.act(out))
        return data



class SheafResidualSAN(nn.Module):
    def __init__(self, num_blocks: int, hidden_dim: int, stalk_dim: int, num_heads:int, dropout: float = 0.2, ablate_sheaves=False, restriction_map_type="low_rank"):
        super().__init__()
        self.num_blocks = num_blocks
        self.ablate_sheaves=ablate_sheaves
        self.hidden_dim = hidden_dim
        self.stalk_dim = stalk_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.restriction_map_type=restriction_map_type
        self.blocks = nn.ModuleList([
            SheafResidualSANBlock(self.hidden_dim, self.stalk_dim, self.num_heads, dropout=self.dropout, ablate_sheaves=self.ablate_sheaves, restriction_map_type=self.restriction_map_type) for _ in range(num_blocks)
        ])

    def forward(self, data):
        for block in self.blocks:
            data = block(data)
        return data



# ----------------------------------------------------------------------------------------------
# this initial dynamics embedding will be the main thing we change when editing the input data
# it will be harder to not hard code some of this stuff
# the same is true for atomic frame embeddings
class InitDynamicsEmbedding(MessagePassing):
    def __init__(self, node_dim, edge_dim, output_dim, num_timesteps=16):
        super().__init__(aggr='mean', node_dim=0)
        self.input_dim = node_dim
        self.edge_dim = edge_dim
        self.output_dim = output_dim
        self.num_timesteps = num_timesteps

        self.mlp = nn.Sequential(
            nn.Linear(edge_dim + 2*node_dim, 64), # TODO don't hard code this, tho it is super specific to the data
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

        self.update_linear = nn.Linear(node_dim + output_dim, output_dim)

        self.distribution_embedding = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

        self.lin_out = nn.Linear(output_dim, output_dim)
        self.gru = nn.GRU(output_dim, output_dim, batch_first=True)

    def forward(self, x, pos, edge_index, edge_attr):

        agg = self.propagate(edge_index, x=x, pos=pos, edge_attr=edge_attr)
        _, h = self.gru(agg)
        return self.lin_out(F.relu(h.squeeze(0))) # use a gru as the overall representation

    def message(self, x_i, x_j, pos_i, pos_j, edge_attr):
        
        dynamics_features = torch.cat([x_i, x_j, edge_attr], dim=-1)

        return self.mlp(dynamics_features)

    def update(self, aggr_out, x):
        return self.update_linear(torch.cat([x, aggr_out], dim=-1))


# the relative frame in this class is inspired by that of PiFold by Gao et al.
class AtomicFrame(nn.Module):
    def __init__(self, atoms, atom_indices, frame_origin="CA"):
        super().__init__()
        self.atoms = atoms
        self.atom_indices = atom_indices
        assert set(self.atoms) == set(self.atom_indices.keys()), "atoms are not the keys of atom_indices"
        assert frame_origin in self.atoms, f"frame origin: {frame_origin} is not in atoms"
        self.frame_origin = frame_origin
        self.register_buffer("not_frame_origin_mask", torch.tensor([atom != self.frame_origin for atom in self.atoms], dtype=torch.bool))
        
        
    def forward(self, x, edge_index, edge_attr):
        carbon_alphas = x[:,:,self.atom_indices[self.frame_origin]]
        u,v = carbon_alphas - x[:,:,self.atom_indices["C"]], x[:,:,self.atom_indices["N"]] - carbon_alphas

        x_basis = u - v
        # normalize
        x_basis = x_basis / torch.clamp(torch.norm(x_basis, dim = -1, keepdim=True), min=0.01)

        y = torch.cross(u, v, dim = -1) 
        # normalize
        y = y / torch.clamp(torch.norm(y, dim=-1, keepdim=True), min=0.01)
        
        # get final orthonormal basis vector:
        z = torch.cross(x_basis,y, dim=-1) # will be normal since other two vectors are normal 
        
        raw_to_basis_matrix = torch.stack([x_basis,y,z], dim=-2) # num_res, n_frames, 3, 3

        # now let's build edge features given by pairwise distances between atoms
        pos_feats = x[:, :, :3*len(self.atom_indices)]          # num_res, n_frames, 3*num_atoms
        edges = torch.stack([pos_feats[edge_index[0]], pos_feats[edge_index[1]]], dim=1)
        edges = edges.view(edge_index.shape[1], 2, x.shape[1], len(self.atom_indices), 3)
        dist_features = torch.cdist(edges[:,0], edges[:,1]) # E, n_frames, len(self.atom_indices), len(self.atom_indices)
        
        # positions will contain the atomic coordinate
        positions = pos_feats[:,:,:len(self.atom_indices)*3].view(x.shape[0], x.shape[1],len(self.atom_indices), 3)

        backbone_centroids = positions.mean(dim=2)
        unpadded = backbone_centroids.diff(dim=1)
        velocities = torch.cat([unpadded[:,:1,:], unpadded],dim=1)

        pair_wise_velocities = F.cosine_similarity(velocities[edge_index[0]], velocities[edge_index[1]], dim=-1).unsqueeze(-1)

        edge_features = torch.cat([dist_features.view(edge_index.shape[1],x.shape[1], len(self.atom_indices)**2), edge_attr[:,None,:].expand(-1,x.shape[1], -1), pair_wise_velocities], dim=2)
 
        # features will be certain positions in the coordinate frame
        relative_features = positions[:, :, self.not_frame_origin_mask, :] - positions[:, :, ~self.not_frame_origin_mask, :] # num_res, n_frames, num_atoms-1, 3
        in_frame_features = torch.matmul(relative_features, raw_to_basis_matrix).view(x.shape[0], x.shape[1], 3*(len(self.atom_indices)-1)) # num_res, n_frames, (num_atoms-1) * 3
        in_frame_velocities = torch.matmul(raw_to_basis_matrix.mT, velocities.unsqueeze(-1)).squeeze(-1)
        
        features = torch.cat([in_frame_features, x[:,:,len(self.atom_indices):], in_frame_velocities], dim=-1)
        
        return features, edge_features # converts from standard I, J, K basis to atomic frame, M^T does the opposite (by definition of orthogonal maps)
        
# -------------------------------------------------------------------------------------------




class NodeSheafClassifier(nn.Module):
    def __init__(self, num_classes=22, hidden_dim=64, num_timesteps=16, stalk_dim=8, num_blocks=8, num_heads=8, ablate_sheaves =False, gat_dropout=0.2, classifier_dropout=0.2,restriction_map_type="low_rank", backbone_atoms=["CA", "N", "C", "O"]):
        super().__init__()

        assert hidden_dim % stalk_dim == 0, "stalk dim must evenly divide hidden dim"

        self.hidden_dim = hidden_dim
        self.stalk_dim = stalk_dim
        self.num_blocks=num_blocks
        self.num_heads=num_heads
        self.num_timesteps=num_timesteps
        self.gat_dropout = gat_dropout
        self.num_classes = num_classes
        self.ablate_sheaves = ablate_sheaves
        self.gat_dropout = gat_dropout
        self.classifier_dropout = classifier_dropout
        self.restriction_map_type = restriction_map_type
        
        self.atomic_frame = AtomicFrame(backbone_atoms, {atom : slice(3 * i, 3 * (i+1)) for i, atom in enumerate(backbone_atoms)}, frame_origin="CA") # atoms are packed left, with dihedral phi, psi and omega on righ

        self.init_dynamics_embedding = InitDynamicsEmbedding(
            25, # 3 componnets of O, C, N's positions in frame space relative to CA, 10 for what?
            18, # 16 for pairwise distances + 1 for bond_edge/not bond_edge
            hidden_dim,
            num_timesteps=self.num_timesteps
        )

        self.label_embedding = nn.Embedding(self.num_classes, self.hidden_dim)

        self.san = SheafResidualSAN(self.num_blocks, self.hidden_dim, self.stalk_dim, self.num_heads, dropout = self.gat_dropout, ablate_sheaves=self.ablate_sheaves, restriction_map_type=self.restriction_map_type)

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.classifier_dropout),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_classes)
        )

    def forward(self, data):
        # take in batch of graphs
        # embed based on egocentric features, since positions are raw and absolute

        data.x, data.edge_attr = self.atomic_frame(data.x, data.edge_index, data.edge_attr)

        data.x = self.init_dynamics_embedding(data.x, data.pos, data.edge_index, data.edge_attr)

        # skip the above if you just want to train on precalced features:

        # add the residue label embedding to unmasked nodes
        data.x = data.x + self.label_embedding(data.y) * data.node_mask[:,None]

        # run sheaf attention
        data = self.san(data)

        data.x = self.classifier(data.x) # classify nodes

        return data







# test if gradients are stable
#if __name__ == "__main__":


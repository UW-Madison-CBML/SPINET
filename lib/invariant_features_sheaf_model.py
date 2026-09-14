import torch
# import torch_sparse
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
from huggingface_hub import PyTorchModelHubMixin
from torch.utils.checkpoint import checkpoint

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
        maps_flat = torch.matmul(self.upper_indices_matrix, upper_tri[:,:,None])

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

class SheafAttentionLearner(nn.Module):
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



class EdgeCrossAttention(nn.Module):
    def __init__(self, input_dim:int, stalk_dim:int):
        super().__init__()
        self.input_dim = input_dim
        self.stalk_dim = stalk_dim
        self.channels = self.input_dim // self.stalk_dim
        self.map_learner = nn.Linear(self.input_dim, self.stalk_dim ** 2)
        self.multihead_attn = nn.MultiheadAttention(self.channels, 1, batch_first=True)
        self.keys = nn.Linear(self.channels, self.channels)
        self.values = nn.Linear(self.channels, self.channels)
        self.queries = nn.Linear(self.channels, self.channels)


    def forward(self, x, edge_index):
        row, col = edge_index

        x_row = x[row].view(-1, self.stalk_dim, self.channels)
        x_col = x[col].view(-1, self.stalk_dim, self.channels)

        attn_output, _ = self.multihead_attn(self.queries(x_row), self.keys(x_col), self.values(x_col))

        return F.relu(attn_output).view(edge_index.shape[1], x.shape[1], self.stalk_dim, self.channels) if len(x.shape) > 3 else F.relu(attn_output)



#-----------------------------------------------------------------------------------
class SheafAttentionConv(MessagePassing):
    def __init__(self, hidden_dim: int, stalk_dim: int, num_heads:int, dropout: float = 0.2, ablate_sheaves=False, restriction_map_type="low_rank", use_attention=True):
        super().__init__(aggr='sum', node_dim=0)
        self.hidden_dim = hidden_dim
        self.stalk_dim = stalk_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_channels = self.hidden_dim // self.stalk_dim
        self.ablate_sheaves = ablate_sheaves
        self.restriction_map_type=restriction_map_type
        self.use_attention = use_attention # this still applies the sheaf across the features channel-wise, it just doesn't calculate alpha

        self.W = nn.Parameter(torch.empty(self.num_heads, self.stalk_dim, self.stalk_dim))
        nn.init.xavier_uniform_(self.W)

        if self.use_attention:
            self.att = nn.Parameter(torch.empty(self.num_heads, 1, 2 * self.hidden_dim))
            nn.init.xavier_uniform_(self.att)

        self.leaky = nn.LeakyReLU(0.2)

        self.update_nodes = nn.Sequential(nn.Linear((self.num_heads + 1) * self.hidden_dim, 4*self.hidden_dim), nn.ReLU(), nn.Linear(4*self.hidden_dim, self.hidden_dim))
        self.update_edges = nn.Sequential(nn.Linear(3 * self.stalk_dim, self.stalk_dim), nn.ReLU(), nn.Linear(self.stalk_dim, self.stalk_dim))

        if not self.ablate_sheaves:
            if self.restriction_map_type == "low_rank":
                self.sheaf_learner = SheafLearnerLowRank(self.hidden_dim, self.stalk_dim, self.stalk_dim//2)
            elif self.restriction_map_type == "orthogonal":
                self.sheaf_learner = SheafLearnerOrthogonal(self.hidden_dim, self.stalk_dim)
            else:
                self.sheaf_learner = SheafLearner(self.hidden_dim, self.stalk_dim)
                



    def forward(self, x, edge_index, edge_attr, return_sheaf=False):
        x_stalk = x.view(x.shape[0], self.num_channels, self.stalk_dim) # this should automatically fail if the stalk_dim is input wrong
        edge_stalk = edge_attr.view(edge_attr.shape[0], self.num_channels, self.stalk_dim)
         
        if not self.ablate_sheaves: 
            maps = self.sheaf_learner(x, edge_attr edge_index)
            _, reverse_edge_indices = sort_edge_index(torch.roll(edge_index,1,0), torch.arange(edge_index.shape[1], device=x.device, dtype=torch.int64))
            neighbor_maps = maps[reverse_edge_indices] 

            i_to_edge = torch.matmul(maps, x_stalk[edge_index[0]].mT).mT
            j_to_edge = torch.matmul(neighbor_maps, x_stalk[edge_index[1]].mT).mT

            edge_forward = self.update_edges(torch.cat([i_to_edge, edge_stalk, j_to_edge], dim=-1)).reshape(edge_attr.shape[0], self.hidden_dim)

            maps = torch.stack([maps, neighbor_maps], dim=1)
            
        else:
            # set transport maps to identity to ablate sheaves
            maps = torch.eye(self.stalk_dim, device=x.device)[None, None, :, :].expand(edge_index.shape[1], 2, -1, -1)
            edge_forward = self.update_edges(torch.cat([x_stalk[edge_index[0]], edge_stalk, x_stalk[edge_index[1]]], dim=-1)).reshape(edge_attr.shape[0], self.hidden_dim)
        
        node_agg = self.propagate(edge_index, x=x, x_stalk=x_stalk, maps=maps, edge_attr=edge_stalk) # num_edges, num_heads, c, d

        if return_sheaf and not self.ablate_sheaves:
            return node_agg, edge_forward, maps[:,0,:,:]
        else:
            return node_agg, edge_forward

    def message(self, x_i, x_j, x_stalk_i, x_stalk_j, maps, edge_attr, index, ptr, size_i):
        
        edge_features = torch.cat([x_i, x_j], dim=-1) 
        edge_features_batched = edge_features[None,:,:].expand(self.num_heads,-1, -1) 
        
        x_stalk_j_batched = x_stalk_j[None,:,:,:].expand(self.num_heads,-1, -1, -1)

        x_stalk_i_batched = x_stalk_i[None,:,:,:].expand(self.num_heads,-1, -1, -1)

        if self.use_attention:
            alpha = self.leaky(torch.einsum("hlf, hef -> hel", self.att, edge_features_batched)).unsqueeze(1)
        else:
            alpha = torch.ones(self.num_heads, 1, x_i.shape[0], 1, device=x_i.device)
        
        transformed_j = torch.einsum("hij, hecj -> heci", self.W, x_stalk_j_batched) #num_heads, num_edges, num_channels, stalk_dim

        if self.use_attention: 
            alpha = softmax(alpha, index, ptr, num_nodes=size_i, dim=2) 
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        if not self.ablate_sheaves: 
            transformed_i = torch.einsum("hij, hecj -> heci", self.W, x_stalk_i_batched)

            i_maps_expanded = maps[None,:,0,:,:].expand(self.num_heads, -1, -1, -1)

            j_maps_expanded = maps[None,:,1,:,:].expand(self.num_heads, -1, -1, -1)
            i_to_edge = torch.matmul(i_maps_expanded, transformed_i.mT)
            j_to_edge = torch.matmul(j_maps_expanded, transformed_j.mT)
            edge_value = edge_attr.mT + (j_to_edge - i_to_edge)
            transported = torch.matmul(i_maps_expanded.mT, edge_value).mT # last mT so that it's c,d not d,c
        else: 
            transported = transformed_j + edge_attr
        alpha = alpha.permute(0,2,1,3).contiguous() # this will end up with num_heads, num_edges, 1, 1 which will broadcast
        return (alpha * transported).permute(1,0,2,3).contiguous() # multiplication by alpha serves as our dropout here

    def update(self, aggr_out, x):
        if(x.ndim > 2):
            x = x.view(x.shape[0], self.hidden_dim)
        if(aggr_out.ndim > 2):
            aggr_out = aggr_out.view(aggr_out.shape[0], self.hidden_dim * self.num_heads)

        return self.update_nodes(torch.cat([x, aggr_out], dim=-1))


    
class SheafResidualSANBlock(nn.Module):
    def __init__(self, hidden_dim: int, stalk_dim: int, num_heads: int, dropout: float = 0.2, ablate_sheaves=False, restriction_map_type="low_rank", use_attention=True):
        super().__init__()
        self.ablate_sheaves = ablate_sheaves
        self.hidden_dim = hidden_dim
        self.stalk_dim = stalk_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.restriction_map_type=restriction_map_type
        self.use_attention = use_attention

        self.san = SheafAttentionConv(self.hidden_dim, self.stalk_dim, self.num_heads, dropout=self.dropout, ablate_sheaves=ablate_sheaves, restriction_map_type=self.restriction_map_type, use_attention=self.use_attention)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.act = nn.ReLU()

    def forward(self, data, return_sheaf=False):
        node_residual = data.x
        edge_residual = data.edge_attr
        if return_sheaf:
            node_out, edge_out, sheaf = self.san(data.x, data.edge_index, data.edge_attr, return_sheaf=return_sheaf)
        else:
            node_out, edge_out = self.san(data.x, data.edge_index, data.edge_attr)
        
        data.x = self.norm(node_residual + self.act(node_out))
        data.edge_attr = self.norm(edge_residual + self.act(edge_out))

        if return_sheaf:
            return data, sheaf
        else:
            return data



class SheafResidualSAN(nn.Module):
    def __init__(self, num_blocks: int, hidden_dim: int, stalk_dim: int, num_heads:int, dropout: float = 0.2, ablate_sheaves=False, restriction_map_type="arbitrary", use_attention=True):
        super().__init__()
        self.num_blocks = num_blocks
        self.ablate_sheaves=ablate_sheaves
        self.hidden_dim = hidden_dim
        self.stalk_dim = stalk_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.restriction_map_type=restriction_map_type
        self.use_attention = use_attention
        self.blocks = nn.ModuleList([
            SheafResidualSANBlock(self.hidden_dim, self.stalk_dim, self.num_heads, dropout=self.dropout, ablate_sheaves=self.ablate_sheaves, restriction_map_type=self.restriction_map_type, use_attention=self.use_attention) for _ in range(num_blocks)
        ])

    def forward(self, data, return_sheaf=False):
        first_sheaf, last_sheaf = None,None 
        for i, block in enumerate(self.blocks):
            if return_sheaf and i == 0:
                data, first_sheaf = block(data, return_sheaf=return_sheaf)
            elif return_sheaf and i == len(self.blocks)-1:
                data, last_sheaf = block(data, return_sheaf=return_sheaf)
            elif return_sheaf:
                data, _ = block(data, return_sheaf=return_sheaf)
            else:
                data = block(data)
        if return_sheaf:
            return data, first_sheaf, last_sheaf
        else:
            return data



# ----------------------------------------------------------------------------------------------
# this initial dynamics embedding will be the main thing we change when editing the input data
# it will be harder to not hard code some of this stuff
# the same is true for atomic frame embeddings
class InitDynamicsEmbedding(MessagePassing):
    def __init__(self, node_dim, edge_dim, hidden_dim, stalk_dim, atoms, atom_indices, frame_origin="CA", restriction_map_type="arbitrary"):
        super().__init__(aggr='sum', node_dim=0)

        self.input_dim = node_dim
        self.edge_dim = edge_dim
        self.stalk_dim = stalk_dim
        self.hidden_dim = hidden_dim
        self.num_channels = self.hidden_dim // self.stalk_dim
        self.atoms = atoms
        self.atom_indices = atom_indices
        assert set(self.atoms) == set(self.atom_indices.keys()), "atoms are not the keys of atom_indices"
        assert frame_origin in self.atoms, f"frame origin: {frame_origin} is not in atoms"
        self.frame_origin = frame_origin
        self.register_buffer("not_frame_origin_mask", torch.tensor([atom != self.frame_origin for atom in self.atoms], dtype=torch.bool))


        self.project_nodes = nn.Linear(self.input_dim, self.hidden_dim)
        self.project_edges = nn.Linear(self.edge_dim, self.hidden_dim)

        self.mlp = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.hidden_dim))

        self.edge_temporal_product = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True) 
        self.node_temporal_product = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True) 

    def forward(self, x, pos, frame_maps, edge_index, edge_attr):
        x = self.project_nodes(x)
        # need to calc some edge features here, since we don't know the edges yet at 

        origin = pos[:, :, ~self.not_frame_origin_mask]
        other_atoms = pos[:, :, self.not_frame_origin_mask]
        in_frame_atoms = torch.matmul(other_atoms[edge_index[1]] - origin[edge_index[0]], frame_maps[edge_index[0]])
        in_frame_atoms = in_frame_atoms.view(edge_index.shape[1], pos.shape[1], 9)
        pairwise_matrices = F.cosine_similarity(frame_maps[edge_index[0]], frame_maps[edge_index[1]], dim=-1) # num_edges, num_times, 3; compare angles of frames
        edge_features = torch.cat([in_frame_atoms, edge_attr[:,None,:].expand(-1,x.shape[1], -1), pairwise_matrices], dim=2)

        # get edge embeddings
        edge_attr = self.project_edges(edge_features)

        _, x = self.node_temporal_product(x)
        _, edge_attr = self.edge_temporal_product(edge_attr)
        x = x.squeeze(0)
        edge_attr = edge_attr.squeeze(0)
        
        return x, edge_attr

class NodeSheafClassifier(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config=None, atoms=["CA", "N", "C", "O"], frame_origin="CA", num_classes=22, hidden_dim=64, num_timesteps=16, stalk_dim=8, num_blocks=8, num_heads=8, ablate_sheaves =False, gat_dropout=0.2, classifier_dropout=0.2,restriction_map_type="low_rank", use_attention=True, use_masking=True):
        super(NodeSheafClassifier, self).__init__()

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
        self.use_attention = use_attention
        self.use_masking = use_masking
        self.atoms = atoms
        self.frame_origin = frame_origin
        self.atom_indices = {atom: slice(3*i,3*(i+1)) for i, atom in enumerate(atoms)}

        #TODO add atoms and frame origin to config
        if(config != None):
            if isinstance(config, dict):
                self.hidden_dim = config.get("hidden_dim", hidden_dim)
                self.stalk_dim = config.get("stalk_dim", stalk_dim)
                self.num_blocks = config.get("num_blocks", num_blocks)
                self.num_heads = config.get("num_heads", num_heads)
                self.num_timesteps = config.get("num_timesteps", num_timesteps)
                self.gat_dropout = config.get("gat_dropout", gat_dropout)
                self.num_classes = config.get("num_classes", num_classes)
                self.ablate_sheaves = config.get("ablate_sheaves", ablate_sheaves)
                self.gat_dropout = config.get("gat_dropout", gat_dropout)
                self.classifier_dropout = config.get("classifier_dropout", classifier_dropout)
                self.restriction_map_type = config.get("restriction_map_type", restriction_map_type)
                self.use_attention = config.get("use_attention", use_attention)
                self.use_masking = config.get("use_masking", use_masking)

            else:
                self.hidden_dim = config.hidden_dim
                self.stalk_dim = config.stalk_dim
                self.num_blocks= config.num_blocks
                self.num_heads= config.num_heads
                self.num_timesteps= config.num_timesteps
                self.gat_dropout = config.gat_dropout
                self.num_classes = config.num_classes
                self.ablate_sheaves = config.ablate_sheaves
                self.gat_dropout = config.gat_dropout
                self.classifier_dropout = config.classifier_dropout
                self.restriction_map_type = config.restriction_map_type
                self.use_attention = config.use_attention
                self.use_masking = config.use_masking
        
        self.init_dynamics_embedding = InitDynamicsEmbedding(12, -1, self.hidden_dim, self.stalk_dim, self.atoms, self.atom_indices, frame_origin=self.frame_origin, ablate_sheaves=self.ablate_sheaves)
        

        self.san = SheafResidualSAN(self.num_blocks, self.hidden_dim, self.stalk_dim, self.num_heads, dropout = self.gat_dropout, ablate_sheaves=self.ablate_sheaves, restriction_map_type=self.restriction_map_type, use_attention=self.use_attention)

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.classifier_dropout),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_classes)
        )

    def forward(self, data, return_sheaf=False):

        data.x, data.edge_attr = self.init_dynamics_embedding(data.x, data.pos, data.frame_maps, data.edge_index, data.edge_attr)

        # run sheaf attention
        if return_sheaf:
            data, first_sheaf, last_sheaf = self.san(data, return_sheaf=return_sheaf)
        else:
            data = self.san(data)

        data.x = self.classifier(data.x) # classify nodes

        if return_sheaf:
            return data, first_sheaf, last_sheaf
        else:
            return data







# test if gradients are stable
#if __name__ == "__main__":


import torch
import torch_sparse
from torch.autograd.gradcheck import gradcheck
from typing import Tuple
from torch_geometric.nn.conv import GATConv
from torch_geometric.nn import MessagePassing
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch.func import functional_call, vmap

#-------------------------------------------------------
# sheaf learners

class SheafLearnerLowRankNormal(nn.Module):
    def __init__(self, in_channels: int, stalk_dim: int, rank: int):
        super().__init__()
        assert rank <= stalk_dim, "Rank 'r' cannot be greater than stalk dimension 'd'."
        self.d = stalk_dim
        self.r = rank
        
        self.q_generator = nn.Linear(in_channels * 2, self.d * self.r)
        self.scale_generator = nn.Linear(in_channels * 2, self.r)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        row, col = edge_index
        edge_features = torch.cat([x[row], x[col]], dim=-1)
        raw_Q = self.q_generator(edge_features).view(-1, self.d, self.r)
        Q, R = torch.linalg.qr(raw_Q) 
        d_sign = torch.diagonal(R, dim1=-2, dim2=-1).sign().unsqueeze(-2)
        Q = Q * d_sign 
        eigenvalues = torch.tanh(self.scale_generator(edge_features)) 
        Sigma = torch.diag_embed(eigenvalues)
        W = torch.matmul(Q, torch.matmul(Sigma, Q.transpose(-1, -2))) 
        return W

class SheafLearnerOrthogonal(nn.Module):
    def __init__(self, input_dim: int, stalk_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.stalk_dim = stalk_dim
        
        self.lin = nn.Linear(self.input_dim * 2, self.input_dim)
        self.triu_learner = nn.Linear(self.input_dim, (self.stalk_dim * (self.stalk_dim-1) )// 2)
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        row, col = edge_index
        edge_features = torch.cat([x[row], x[col]], dim=-1)
        upper_tri = self.triu_learner(F.relu(self.lin(edge_features)))
        maps = torch.zeroes((len(edge_index), self.stalk_dim, self.stalk_dim), device=x.device) 
        upper_indices = torch.triu_indices(self.stalk_dim, self.stalk_dim, 1, device=x.device)# [None,:,:].repeat(len(edge_index), 1, 1)
        lower_indices = torch.tril_indices(self.stalk_dim, self.stalk_dim, -1, device=x.device)
        maps[:, upper_indices] = upper_tri
        maps[:, lower_indices] = -1 * upper_tri
        maps = torch.linalg.matrix_exp(maps) # this forces it into SO(n) for whatever reason
        return maps

class SheafLearner(nn.Module):
    def __init__(self, input_dim:int, stalk_dim:int):
        super().__init__()
        self.input_dim = input_dim
        self.stalk_dim = stalk_dim
        self.lin = nn.Linear(2*self.stalk_dim, 2*self.stalk_dim) # TODO is this too much?
        self.map_learner = nn.Linear(2*self.stalk_dim, self.stalk_dim ** 2)
        self.act = F.relu

    def forward(self, x, edge_index):
        row, col = edge_index

        x_row = x[row]
        x_col = x[col]

        maps = self.map_learner(self.act(self.lin(torch.cat([x_row, x_col], dim=-1)))).view(len(edge_index), self.stalk_dim, self.stalk_dim)

        return maps


#-----------------------------------------------------------------------------------



class SheafAttentionConv(MessagePassing):
    def __init__(self, hidden_dim: int, stalk_dim: int, num_heads:int, dropout: float = 0.2):
        super().__init__(aggr='add', node_dim=0)
        self.hidden_dim = hidden_dim
        self.stalk_dim = stalk_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_channels = self.hidden_dim // self.stalk_dim

        self.W = nn.ModuleList(*[nn.Linear(self.stalk_dim, self.stalk_dim, bias=False) for _ in range(self.num_heads)])
        self.W_params = {
            "weight": torch.stack([layer.weight for layer in self.W]),
        }
        self.att = nn.ModuleList(*[nn.Linear(2 * self.hidden_dim, 1, bias=False) for _ in range(self.num_heads)])
        self.att_params = {
            "weight": torch.stack([layer.weight for layer in self.att]),
        }
        self.project_concat = nn.Linear(self.num_heads * self.hidden_dim, self.hidden_dim)
        self.leaky = nn.LeakyReLU(0.2)

        self.sheaf_learner = SheafLearner(self.hidden_dim, self.stalk_dim)
        
        self.apply_W = vmap(lambda tensor: functional_call(nn.Linear(self.stalk_dim, self.stalk_dim, bias=False), self.W_params, tensor), in_dims=(0, None))
        self.apply_att = vmap(lambda tensor: functional_call(nn.Linear(2 * hidden_dim, 1, bias=False), self.att_params, tensor), in_dims=(0, None))

    def forward(self, x, edge_index):
        maps = self.sheaf_learner(x, edge_index)      
        x_stalk = x.view(edge_index, self.num_channels, self.stalk_dim) # this should automatically fail if the stalk_dim is input wrong
        out = self.propagate(edge_index, x=x, x_stalk=x_stalk, maps=maps)
        return self.project_concat(torch.cat(out, dim=-1))

    def message(self, x_i, x_j, x_stalk_j, maps, index, ptr, size_i):

        alpha = self.leaky(self.apply_att(torch.cat([x_i, x_j], dim=-1))) # num_heads, num_edges, 1?
        
        alpha = softmax(alpha, index, ptr, size_i)                  
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        transformed = self.apply_W(x_stalk_j) # num_heads, num_edges, stalk_dim
        transported = torch.bmm(maps, transformed.unsqueeze(-1)).squeeze(-1)

        return alpha * transported


class SheafResidualSANBlock(nn.Module):
    def __init__(self, hidden_dim: int, stalk_dim: int, num_heads: int, dropout: float = 0.2):
        super().__init__()
        self.san = SheafAttentionConv(hidden_dim, stalk_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU()

    def forward(self, data):
        residual = data.x
        out = self.san(data.x, data.edge_index)
        data.x = self.norm(residual + self.act(out))
        return data



class SheafResidualSAN(nn.Module):
    def __init__(self, num_blocks: int, hidden_dim: int, stalk_dim: int, num_heads:int, dropout: float = 0.2):
        super().__init__()
        self.blocks = nn.ModuleList([
            SheafResidualSANBlock(hidden_dim, stalk_dim, num_heads, dropout=dropout) for _ in range(num_blocks)
        ])

    def forward(self, data):
        for block in self.blocks:
            data = block(data)
        return data

class MultiStalkLaplacian(nn.Module):
    def __init__(self, num_groups: int, stalk_dim: int):
        super().__init__()
        self.g = num_groups
        self.d = stalk_dim
        self.laplacians = nn.ModuleList(
            [SheafLaplacian(stalk_dim) for _ in range(num_groups)]
        )
        self.learners = nn.ModuleList(
            [SheafLearnerLowRankNormal(num_groups * stalk_dim, stalk_dim, stalk_dim // 2)
             for _ in range(num_groups)]
        )

    def forward(self, data):
        x_groups = data.x.view(data.x.size(0), self.g, self.d)
        outs = []
        for k in range(self.g):
            sub = data.clone()
            sub.x = x_groups[:, k, :]
            sub.maps = self.learners[k](data.x, data.edge_index)  # condition on full feature, not just the slice
            sub = self.laplacians[k](sub)
            outs.append(sub.x)
        data.x = torch.cat(outs, dim=-1)
        return data
 

class SheafLaplacian(nn.Module):
    def __init__(self, stalk_dim: int):
        super().__init__()
        self.d = stalk_dim

    def forward(self, data):
        num_nodes = data.x.size(0)
        device = data.x.device
        
        row, col = data.edge_index
        num_edges = data.edge_index.size(1)
        
        w_t_w = torch.matmul(data.maps.transpose(-1, -2), data.maps)
        off_diag_values = -w_t_w
        
        grid_x, grid_y = torch.meshgrid(torch.arange(self.d, device=device), torch.arange(self.d, device=device), indexing='ij')
        
        off_diag_rows = (col.unsqueeze(1).unsqueeze(2) * self.d + grid_x).flatten()
        off_diag_cols = (row.unsqueeze(1).unsqueeze(2) * self.d + grid_y).flatten()
        off_diag_indices = torch.stack([off_diag_rows, off_diag_cols], dim=0)
        off_diag_flat_values = off_diag_values.flatten()

        diag_values = torch.zeros(num_nodes, self.d, self.d, device=device)
        diag_values.index_add_(0, row, w_t_w)
        
        node_indices = torch.arange(num_nodes, device=device)
        diag_rows = (node_indices.unsqueeze(1).unsqueeze(2) * self.d + grid_x).flatten()
        diag_cols = (node_indices.unsqueeze(1).unsqueeze(2) * self.d + grid_y).flatten()
        diag_indices = torch.stack([diag_rows, diag_cols], dim=0)
        diag_flat_values = diag_values.flatten()

        all_indices = torch.cat([off_diag_indices, diag_indices], dim=1)
        all_values = torch.cat([off_diag_flat_values, diag_flat_values], dim=0)
        matrix_dim = num_nodes * self.d
        L_F_sparse = torch.sparse_coo_tensor(all_indices, all_values, (matrix_dim, matrix_dim)).coalesce()
        x_global = data.x.view(matrix_dim, 1)
        laplacian_product = torch.sparse.mm(L_F_sparse, x_global)
        out = x_global - laplacian_product
        data.x = out.view(num_nodes, self.d)
        return data


class SheafResidualGATBlock(nn.Module):  
    def __init__(self, num_heads, hidden_dim, dropout=0.2, ablate_sheaves=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.ablate_sheaves = ablate_sheaves 
        self.num_heads = num_heads
        if(not self.ablate_sheaves):
            # 1 sheaf learner per block
            self.sheaf_learner = SheafLearnerLowRankNormal(self.hidden_dim, self.hidden_dim, self.hidden_dim // 4) # alternatively SheafLearner(self.hidden_dim, self.hidden_dim)
            self.apply_laplacian = SheafLaplacian(self.hidden_dim)
        self.gat_block = GATConv(self.hidden_dim, self.hidden_dim, heads=self.num_heads, concat=False, dropout=self.dropout) # TODO check how this is implemented

    def forward(self, data):
        # in case of custom residual definition: skip = data.x
        if(not self.ablate_sheaves):
            data.maps = self.sheaf_learner(data.x, data.edge_index)

        data.x = self.gat_block(data.x, data.edge_index)

        if(not self.ablate_sheaves):
            data = self.apply_laplacian(data)

        return data
        

        
class SheafResidualGAT(nn.Module):  
    def __init__(self, num_blocks, num_heads, hidden_dim, dropout=0.2, ablate_sheaves=False):
        super().__init__()
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.ablate_sheaves = ablate_sheaves 

        self.blocks = nn.Sequential(
            *[SheafResidualGATBlock(self.num_heads, self.hidden_dim, dropout=self.dropout, ablate_sheaves=self.ablate_sheaves) for _ in range(self.num_blocks)]
        ) 

    def forward(self, data):
        return self.blocks(data)

# ----------------------------------------------------------------------------------------------
# this  initial dynamics embedding will be the main thing we change when editing the input data
# it will be harder to not hard code some of this stuff
class InitDynamicsEmbedding(MessagePassing):  
    def __init__(self, output_dim, input_dim = 5, velocity_range = (0,3)):
        super().__init__(aggr='mean') 
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.velocity_range = velocity_range
        
        self.mlp = nn.Sequential(
            nn.Linear(15, 16), # TODO don't hard code this, tho it is super specific to the data
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )
        
        self.update_linear = nn.Linear(input_dim + output_dim, output_dim)

    def forward(self, x, pos, edge_index):
        return self.propagate(edge_index, x=x, pos=pos)

    def message(self, x_i, x_j, pos_i, pos_j):
        relative_pos = pos_j - pos_i
        non_vel_features_x_i = x_i[:, self.velocity_range[1]:] # non velocity features like bond_ang, bond_len
        non_vel_features_x_j = x_j[:, self.velocity_range[1]:]
        d_pos_i = x_i[:,self.velocity_range[0]:self.velocity_range[1]] # these are velocity
        d_pos_j = x_j[:,self.velocity_range[0]:self.velocity_range[1]] # 
        prev_pos_i = pos_i - d_pos_i
        prev_pos_j = pos_j - d_pos_j
        
        d_pos_j_angle = F.cosine_similarity(prev_pos_j - prev_pos_i, d_pos_j, dim=-1).unsqueeze(-1)
        d_pos_i_angle = F.cosine_similarity(prev_pos_j - prev_pos_i, d_pos_i, dim=-1).unsqueeze(-1)

        dynamics_features = torch.cat([relative_pos, non_vel_features_x_i, non_vel_features_x_j, d_pos_i, d_pos_j, d_pos_i_angle, d_pos_j_angle], dim=-1)
         
        return self.mlp(dynamics_features)

    def update(self, aggr_out, x):
        return self.update_linear(torch.cat([x, aggr_out], dim=-1))        
# -------------------------------------------------------------------------------------------
         
 


class NodeSheafGATClassifier(nn.Module):  
    def __init__(self, num_classes=22, hidden_dim=64, stalk_dim=8, num_blocks=8, num_heads=8, ablate_sheaves =False, gat_dropout=0.2, classifier_dropout=0.2):
        super().__init__()

        assert hidden_dim % stalk_dim == 0, "stalk dim must evenly divide hidden dim"

        self.hidden_dim = hidden_dim
        self.stalk_dim = stalk_dim
        self.num_blocks=num_blocks
        self.num_heads=num_heads
        self.gat_dropout = gat_dropout
        self.num_classes = num_classes
        self.ablate_sheaves = ablate_sheaves
        self.gat_dropout = gat_dropout
        self.classifier_dropout = classifier_dropout

        self.init_dynamics_embedding = InitDynamicsEmbedding(
            hidden_dim,
            input_dim = 5,
            velocity_range = (0,3), # exclusive on right, other features will be already egocentric
        )

        self.label_embedding = nn.Embedding(self.num_classes, self.hidden_dim) 

        #self.sheaf_residual_gat = SheafResidualGAT(self.num_blocks, self.num_heads, self.hidden_dim, self.stalk_dim, dropout=self.gat_dropout, ablate_sheaves=self.ablate_sheaves) 
        self.san = SheafResidualSAN(self.num_blocks, self.hidden_dim, self.stalk_dim, self.num_heads, dropout = self.gat_dropout)
                
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
        data.x = self.init_dynamics_embedding(data.x, data.pos, data.edge_index)

        # add the residue label embedding to unmasked nodes
        data.x = data.x + self.label_embedding(data.y) * data.node_mask[:,None]

        # run sheaf gat residual blocks
        #data = self.sheaf_residual_gat(data)
        data = self.san(data)

        data.x = self.classifier(data.x) # classify nodes

        return data
        
        


        
        
        
# test if gradients are stable     
#if __name__ == "__main__":
        

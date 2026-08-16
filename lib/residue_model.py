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
from torch.nn.utils.rnn import pack_padded_sequence

# this model directly uses positions at MD timesteps instead of relative features
# combines time-series positions with LSTM to node embedding
# then learns sheaf over these embeddings and applies the Laplacian

#-------------------------------------------------------
# sheaf learners

class NormalizedBlockLaplacian(nn.Module):
    def __init__(self, stalk_dim: int, eps: float = 1e-5):
        """
        Build symmetrically normalized block Laplacian 
        from the restriction maps of any sheaf learner.
        """
        super().__init__()
        self.d = stalk_dim
        self.eps = eps

    def forward(self, maps: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        device = maps.device
        row, col = edge_index
        
        # A_uv = W^T W (The off-diagonal blocks before normalization)
        w_t_w = torch.matmul(maps.transpose(-1, -2), maps)
        
        # 1. Compute Degree matrices D_v
        diag_values = torch.zeros(num_nodes, self.d, self.d, device=device)
        diag_values.index_add_(0, row, w_t_w)
        
        # 2. Compute D^{-1/2} block-wise via eigendecomposition
        I = torch.eye(self.d, device=device).unsqueeze(0)
        D_reg = diag_values + self.eps * I
        
        eigenvalues, eigenvectors = torch.linalg.eigh(D_reg)
        eigenvalues = torch.clamp(eigenvalues, min=self.eps)
        inv_sqrt_eigenvalues = torch.diag_embed(1.0 / torch.sqrt(eigenvalues))
        D_inv_sqrt = eigenvectors @ inv_sqrt_eigenvalues @ eigenvectors.transpose(-1, -2)
        
        # 3. Normalize off-diagonal blocks: -D_v^{-1/2} A_uv D_u^{-1/2}
        off_diag_values = -torch.bmm(D_inv_sqrt[col], torch.bmm(w_t_w, D_inv_sqrt[row]))
        norm_diag_values = I.repeat(num_nodes, 1, 1)

        # 4. Construct Sparse Matrix Indices
        grid_x, grid_y = torch.meshgrid(
            torch.arange(self.d, device=device), 
            torch.arange(self.d, device=device), 
            indexing='ij'
        )
        
        # Off-diagonal indices
        off_diag_rows = (col.unsqueeze(1).unsqueeze(2) * self.d + grid_x).flatten()
        off_diag_cols = (row.unsqueeze(1).unsqueeze(2) * self.d + grid_y).flatten()
        off_diag_indices = torch.stack([off_diag_rows, off_diag_cols], dim=0)

        # Diagonal indices
        node_indices = torch.arange(num_nodes, device=device)
        diag_rows = (node_indices.unsqueeze(1).unsqueeze(2) * self.d + grid_x).flatten()
        diag_cols = (node_indices.unsqueeze(1).unsqueeze(2) * self.d + grid_y).flatten()
        diag_indices = torch.stack([diag_rows, diag_cols], dim=0)

        # 5. Assemble Sparse Tensor
        all_indices = torch.cat([off_diag_indices, diag_indices], dim=1)
        all_values = torch.cat([off_diag_values.flatten(), norm_diag_values.flatten()], dim=0)
        matrix_dim = num_nodes * self.d
        
        L_sym_sparse = torch.sparse_coo_tensor(
            all_indices, all_values, (matrix_dim, matrix_dim)
        ).coalesce()
        
        return L_sym_sparse

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
        num_edges = edge_index.size(1)
        edge_features = torch.cat([x[row], x[col]], dim=-1)
        upper_tri = self.triu_learner(F.relu(self.lin(edge_features)))
        maps = torch.zeros((num_edges, self.stalk_dim, self.stalk_dim), device=x.device) 
        upper_indices = torch.triu_indices(self.stalk_dim, self.stalk_dim, 1, device=x.device)# [None,:,:].repeat(len(edge_index), 1, 1)
        lower_indices = torch.tril_indices(self.stalk_dim, self.stalk_dim, -1, device=x.device)
        maps[:, upper_indices[0], upper_indices[1]] = upper_tri
        maps[:, lower_indices[0], lower_indices[1]] = -1 * upper_tri

        maps = torch.linalg.matrix_exp(maps) # this forces it into SO(n) for whatever reason
        return maps

class SheafLearner(nn.Module):
    def __init__(self, input_dim:int, stalk_dim:int):
        super().__init__()
        self.input_dim = input_dim
        self.stalk_dim = stalk_dim

        self.lin = nn.Linear(2*self.input_dim, 2*self.stalk_dim) # TODO is this too much?
        self.map_learner = nn.Linear(2*self.stalk_dim, self.stalk_dim ** 2)
        self.act = F.relu

    def forward(self, x, edge_index):
        row, col = edge_index

        x_row = x[row]
        x_col = x[col]

        maps = self.map_learner(self.act(self.lin(torch.cat([x_row, x_col], dim=-1)))).view(len(row), self.stalk_dim, self.stalk_dim)

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
        self.restriction_map_type = restriction_map_type

        self.W_weights = nn.Parameter(torch.empty(self.num_heads, self.stalk_dim, self.stalk_dim))
        self.att_weights = nn.Parameter(torch.empty(self.num_heads, 1, 2 * self.hidden_dim))
        nn.init.xavier_uniform_(self.W_weights)
        nn.init.xavier_uniform_(self.att_weights)

        self.project_concat = nn.Linear(self.num_heads * self.hidden_dim, self.hidden_dim)
        self.leaky = nn.LeakyReLU(0.2)
        if not self.ablate_sheaves:
            if self.restriction_map_type == "low_rank":
                self.sheaf_learner = SheafLearnerLowRankNormal(self.hidden_dim, self.stalk_dim, self.stalk_dim//2)
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
        edge_features = torch.cat([x_i, x_j], dim=-1) # num_edges, 2 * hidden_dim

        # calculate attention
        raw_alpha = torch.einsum('h i d, e d -> h e i', self.att_weights, edge_features)
        alpha = self.leaky(raw_alpha)
        
        # apply attention
        transformed = torch.einsum('h s t, e c t -> h e c s', self.W_weights, x_stalk_j)
        
        # softmax over edges
        alpha = softmax(alpha, index, ptr, num_nodes=size_i, dim=1) 
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        if not self.ablate_sheaves: 
            maps_expanded = maps[None,:,:,:].expand(self.num_heads, -1, -1, -1)
            transported = torch.matmul(maps_expanded, transformed.mT)
        else: 
            transported = transformed

        alpha = alpha.unsqueeze(-1) # this will end up with num_heads, num_edges, 1, 1 which will broadcast

        return (alpha * transported).permute(1,0,2,3).contiguous()
    

        
class SheafResidualSANBlock(nn.Module):
    def __init__(self, hidden_dim: int, stalk_dim: int, num_heads: int, dropout: float = 0.2, ablate_sheaves = False, restriction_map_type = "low_rank"):
        super().__init__()
        self.ablate_sheaves = ablate_sheaves
        self.hidden_dim = hidden_dim
        self.stalk_dim = stalk_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.restriction_map_type = restriction_map_type

        self.san = SheafAttentionConv(self.hidden_dim, self.stalk_dim, self.num_heads, dropout=self.dropout, ablate_sheaves=ablate_sheaves, restriction_map_type=self.restriction_map_type)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU()

    def forward(self, data):
        residual = data.x
        out = self.san(data.x, data.edge_index)
        data.x = self.norm(residual + self.act(out))
        return data



class SheafResidualSAN(nn.Module):
    def __init__(self, num_blocks: int, hidden_dim: int, stalk_dim: int, num_heads : int, dropout: float = 0.2, ablate_sheaves = False, restriction_map_type = "low_rank"):
        super().__init__()
        self.num_blocks = num_blocks
        self.ablate_sheaves = ablate_sheaves
        self.hidden_dim = hidden_dim
        self.stalk_dim = stalk_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.restriction_map_type = restriction_map_type
        self.blocks = nn.ModuleList([
            SheafResidualSANBlock(self.hidden_dim, self.stalk_dim, self.num_heads, dropout=self.dropout, ablate_sheaves=self.ablate_sheaves, restriction_map_type=self.restriction_map_type) for _ in range(num_blocks)
            ])

    def forward(self, data):
        for block in self.blocks:
            data = block(data)
        return data

# ----------------------------------------------------------------------------------------------
# 1D-CNN that encodes MD trajectories
class DynamicsTrajectoryEmbedding(nn.Module):  
    def __init__(self, input_size, hidden_size, kernel_size=3, padding=1, embedding_dim=16):
        super().__init__() 
        
        self.conv1 = nn.Conv1d(
            in_channels = input_size,
            out_channels = hidden_size,
            kernel_size = kernel_size,
            padding = padding
        )
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.conv2 = nn.Conv1d(
                in_channels = hidden_size,
                out_channels = hidden_size,
                kernel_size = kernel_size,
                padding = padding
                )
        self.bn2 = nn.BatchNorm1d(hidden_size)
        
        self.fc = nn.Linear(hidden_size, embedding_dim)
        self.act = nn.ReLU()

    def forward(self, x, seq_lengths):
        # x shape: (num_nodes, max_seq_len, features)
        # seq_lengths shape: (num_nodes,) - true integer lengths

        x = x.permute(0, 2, 1) # batch, channels, length

        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))

        max_len = x.size(2)
        node_indices = torch.arange(max_len, device=x.device).unsqueeze(0)
        mask = node_indices >= seq_lengths.unsqueeze(1)

        x = x.masked_fill(mask.unsqueeze(1), float('-inf'))

        x_pooled, _ = torch.max(x, dim=2)

        embedding = self.fc(x_pooled)

        return embedding

# -------------------------------------------------------------------------------------------

class NodeSheafAttentionClassifier(nn.Module):  
    def __init__(self, input_size, num_classes=20, hidden_dim=64, stalk_dim=16, num_lstm_layers=1, num_blocks=8, num_heads=8, ablate_sheaves=False, gat_dropout=0.2, classifier_dropout=0.2):
        super().__init__()

        assert hidden_dim % stalk_dim == 0, "stalk dim must evenly divide hidden dim"

        self.hidden_dim = hidden_dim
        self.stalk_dim = stalk_dim
        self.num_lstm_layers = num_lstm_layers
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.gat_dropout = gat_dropout
        self.num_classes = num_classes
        self.ablate_sheaves = ablate_sheaves
        self.gat_dropout = gat_dropout
        self.classifier_dropout = classifier_dropout

        self.dynamics_trajectory_embedding = DynamicsTrajectoryEmbedding(
            input_size = input_size,
            hidden_size = hidden_dim,
            embedding_dim = hidden_dim
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

        # expand from (batch_size,) to (total_nodes,)
        # LSTM treats every node as a sequence
        node_seq_lengths = data.lengths[data.batch]

        # take in batch of graph trajectories
        # embed based on time-series trajectories per node
        # input data.pos for trajectories instead of data.x
        data.x = self.dynamics_trajectory_embedding(data.pos, node_seq_lengths)

        # add the residue label embedding to unmasked nodes
        data.x = data.x + self.label_embedding(data.y) * data.node_mask[:,None]

        # run sheaf gat residual blocks
        #data = self.sheaf_residual_gat(data)
        data = self.san(data)
        data.x = self.classifier(data.x) # classify nodes

        return data
        
        


        
        
        
# test if gradients are stable     
#if __name__ == "__main__":
        

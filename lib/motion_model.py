import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from laplacian import sheaf_laplacian, sheaf_laplacian_adjacency
from sheaf_utils import eigenspectrum, eigenvectors
from torch.autograd.gradcheck import gradcheck
# TODO add hugging face pytorchmixin
class SheafMotionClassifier(torch.nn.Module):  
    def __init__(self, node_features, stalk_dimensions, K=8, lstm_hidden_dim=8, num_classes=5, hidden_dim=64, adjacency_matrix=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_features = node_features
        self.stalk_dimensions = stalk_dimensions 
        # bool: True if using adj mat, False if using edge list
        self.adjacency_matrix = adjacency_matrix
        # MOTION_CLASSES = ["PE","PS","PF","PC","OM"]
        self.num_classes = num_classes
        self.lstm_hidden_dim = lstm_hidden_dim

        # apply to the nodes
        self.lin1 = torch.nn.Linear(self.node_features, self.hidden_dim)
        self.lin2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)

        # Bound activations before Laplacian solver.
        self.norm1 = torch.nn.LayerNorm(self.hidden_dim)
        self.norm2 = torch.nn.LayerNorm(self.hidden_dim)

        # apply to the ordered pairs of node hidden features
        self.lin3 = torch.nn.Linear(self.hidden_dim*2, self.stalk_dimensions**2)
        # each lstm looks at 4 features: the complex and real parts of the two eigvals from the two proteins

        # First K eigenvectors extracted for constant input dimension for LSTM
        self.K = K
        #self.lstm = torch.nn.LSTM(self.K, self.lstm_hidden_dim, batch_first=True, bidirectional=True)
        self.covariance_processor = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(self.K * self.K, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, self.lstm_hidden_dim * 2) # Matches your current lin4 input
        ) # Try without LSTM for now, as the input is constant size and doesn't need sequence processing

        self.lin4 = torch.nn.Linear(self.lstm_hidden_dim*2, self.lstm_hidden_dim*2)
        self.lin5 = torch.nn.Linear(self.lstm_hidden_dim*2, self.num_classes)
        
    def forward(self, nodes,  node_lengths, matrix=None, edges=None, edge_lengths=None):
        # T = num_nodes
        # E = num_edges
        # nodes: shape = (B, 2, T, N)
        # node_lengths: shape = (B), type = int, 0 <= min, max < T
        # matrix: shape = B, 2, T, T, type = bool
        # edges: shape = B, 2, E, 2
        # edge_lengths = (B), type = int, 0 <= min, max < E
        if(not self.adjacency_matrix and (edges is None or edge_lengths is None)):
            raise ValueError("must provide edges and edge padding if not using adjacency matrices")
        if(self.adjacency_matrix and matrix is None):
            raise ValueError("must provide matrix if using adjacency matrices")

        #B,2,T,self.node_features
        B,_, T,N = nodes.shape

        nodes = F.relu(self.norm1(self.lin1(nodes)))
        nodes = F.relu(self.norm2(self.lin2(nodes))) # B,2,T,hidden_dim

        if(not self.adjacency_matrix):
            # B, 2, E, 2
            _,_,E,_ = edges.shape
            # get the actual graphs 
            #TODO implement differentiable indexing here 
            left_graphs = nodes[torch.arange(B), torch.arange(2)[None,:].repeat(B,1), edges[:,:,:,0]]
            right_graphs = nodes[torch.arange(B), torch.arange(2)[None,:].repeat(B,1), edges[:,:,:,1]]
            graphs = torch.stack([left_graphs,right_graphs], dim=3) # B,2,E,2,hidden_dim
            graphs = torch.cat([graphs, graphs.roll(3,1)], dim=2) # B,2,2*E,2,hidden_dim
        
            graphs = graphs.reshape(B,2,2*E,2*self.hidden_dim)
            sheaves = F.relu(self.lin3(graphs)) #B,2,2*E,stalk_dim^2
            # reshape the batches of two sheaves for each conformation into the batches dimension
            sheaves = sheaves.reshape(B*2,E,2,self.stalk_dimensions, self.stalk_dimensions)
            edges = edges.reshape(B*2, E, 2)
            _, eigvects = eigenvectors(*sheaf_laplacian(sheaves,edges,node_lengths)).reshape(B,2,T,T) # B,2,T,T
        else:
            node_pairs = torch.cat(torch.broadcast_tensors(nodes[:,:,:,None,:],nodes[:,:,None,:,:]), dim=4) # B,2,T,T,2*hidden_dim
            

            # now mask the sheaves
            node_pairs = matrix[:,:,:,:,None] * node_pairs

            flat_sheaves = self.lin3(node_pairs) # B, 2, T, T, D**2

            sheaves = flat_sheaves.reshape(B,2,T,T,self.stalk_dimensions,self.stalk_dimensions)
            # flatten out pair dim
            sheaves = sheaves.reshape(B*2,T,T,self.stalk_dimensions,self.stalk_dimensions)
            print(sheaves)
            # node lengths needs to be doubled for the flattened pair dim
            node_lengths = node_lengths.to(sheaves.device)
            # D = self.stalk_dimensions
            laps, lap_lens = sheaf_laplacian_adjacency(sheaves,node_lengths[:,None].repeat(1,2).flatten())
            print(laps)
            _, eigvects = eigenvectors(laps, lap_lens) # complex
            print("eig:", eigvects)
            eigvects = eigvects.reshape(B,2,T*self.stalk_dimensions,T*self.stalk_dimensions)

        # Truncate to 1st K eigenvectors
        U1_k = eigvects[:, 0, :, :self.K] #(B, T*D, K)
        U2_k = eigvects[:, 1, :, :self.K] #(B, T*D, K)

        # Pad with zeros if the graph has fewer than K eigenvectors
        actual_k = U1_k.shape[-1]
        if actual_k < self.K:
            padding = torch.zeros(B, U1_k.shape[1], self.K - actual_k, device=U1_k.device, dtype=U1_k.dtype)
            U1_k = torch.cat([U1_k, padding], dim=-1)
            U2_k = torch.cat([U2_k, padding], dim=-1)

        # Compute K x K cross-covariance matrix
        C_k = torch.bmm(U1_k.transpose(1,2), U2_k) #(B, K, K)

        # No packing needed since K is constant
        h = self.covariance_processor(C_k)
        #h = h.permute(1, 2, 0).reshape(B, 2 * self.lstm_hidden_dim)
        pair_features = F.relu(self.lin4(h))
        out = self.lin5(pair_features)
        return out
        
        
        
        


        
        
        
# test if gradients are stable     
if __name__ == "__main__":
    from torchinfo import summary
    B = 2
    T = 300
    E = 2 

    # node_features, stalk_dimensions, lstm_hidden_dim=8, num_classes=5, hidden_dim=64, adjacency_matrix=True
    model = SheafMotionClassifier(1, 1, lstm_hidden_dim=8, num_classes=5, hidden_dim=8, adjacency_matrix=True).double()
    nodes = torch.randn(B, 2, T, 1, requires_grad=True).to(torch.double)
    node_lengths = torch.tensor([T]*B, dtype=torch.int)

    matrix_first_half = (torch.eye(T, dtype=torch.bool).roll(0,1) | torch.eye(T, dtype=torch.bool).roll(1,1))[None,None,:,:].repeat(B//2,2,1,1)
    matrix_second_half = torch.zeros(T,T,dtype=torch.bool)
    matrix_second_half[0,2] = 1
    matrix_second_half[2,0] = 1
    matrix_second_half = matrix_second_half[None,None,:,:].repeat(B//2,2,1,1)
    matrix = torch.cat([matrix_first_half, matrix_second_half],dim=0)

    summary(model, input_data=(nodes, node_lengths, matrix)) 
    

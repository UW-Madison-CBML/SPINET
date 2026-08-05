import torch
import torch.nn.functional as F
import torch_sparse
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd.gradcheck import gradcheck
from typing import Tuple

import diffusion_laplace as lap


class SheafLearner(torch.nn.Module):
    """Learns a sheaf from local features and stalk dimensions."""
    def __init__(self, in_channels: int, out_shape: Tuple[int, int]):
        super().__init__()
        self.out_shape = out_shape
        # The output needs to form two d x d matrices per edge (source and target)
        self.linear1 = torch.nn.Linear(in_channels, 2 * out_shape[0] * out_shape[1])
        self.act = torch.tanh 

    def forward(self, X, edge_index):
        # X: shape = (num_total_nodes, num_features)
        # edge_index: shape = (2, num_edges)
        row, col = edge_index

        x_row = X[row]
        x_col = X[col]

        # Concatenate local features and pass through linear layer + activation
        maps = self.linear1(torch.cat([x_row, x_col], dim=-1))
        maps = self.act(maps)
        
        # Reshape to (num_edges, 2, d, d)
        maps = maps.view(-1, 2, self.out_shape[0], self.out_shape[1])

        return maps



class NodeSheafGATClassifier(torch.nn.Module):  
    def __init__(self, num_classes=22, hidden_dim=64, num_blocks=8, num_heads=8, ablate_sheaves =False, gat_dropout=0.2, classifier_dropout=0.2):
        super().__init__()

        self.stalk_dimensions = 3 
        self.hidden_dim = hidden_dim
        self.num_blocks=num_blocks
        self.num_heads=num_heads
        self.dropout = dropout
        self.num_classes = num_classes
        self.ablate_sheaves = ablate_sheaves
        self.gat_dropout = gat_dropout
        self.classifier_dropout = classifier_dropout

        self.init_dynamics_embedding = InitDynamicsEmbedding(
            velocity_range = (0,3), # exclusive, other features will be already egocentric
            output_dim = hidden_dim  
            
        )
        self.label_embedding = torch.nn.Embedding(self.num_classes, self.hidden_dim) 

        self.sheaf_residual_gat = SheafResidualGAT(self.num_blocks, self.num_heads, self.hidden_dim, dropout=self.gat_dropout, ablate_sheaves=self.ablate_sheaves) 

                
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.classifier_dropout),
            torch.nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, self.num_classes)
        )
        
    def forward(self, data):
        # take in batch of graphs
        # embed based on egocentric features, since positions are raw and absolute
        data = self.init_dynamics_embedding(data)

        # add the residue label embedding to unmasked nodes
        data.x = data.x + self.label_embedding(data.y) * data.node_mask[:,None]

        # run sheaf gat residual blocks
        data = self.sheaf_residual_gat(data)

        data.x = self.classifier(data.x) # classify nodes

        return data
        
        


        
        
        
# test if gradients are stable     
if __name__ == "__main__":
   # ==========================================
    # 1. Define Dummy Dimensions
    # ==========================================
    B = 4            # Batch size
    num_confs = 2    # Number of conformations
    N = 15           # Number of nodes per graph
    F_dim = 8        # Number of node features
    E = 20           # Number of edges per graph
    num_classes = 5  # Motion classes
    hidden_dim = 16  # Internal representation size

    print("--- Initializing Dummy Data ---")
    # X: (B, 2, N, F)
    X_dummy = torch.randn(B, num_confs, N, F_dim, requires_grad=True)
    
    # edge_index: (B, 2, 2, E)
    # Node indices must be valid integers between 0 and N-1
    edge_index_dummy = torch.randint(0, N, (B, num_confs, 2, E))
    
    print(f"X shape: {X_dummy.shape}")
    print(f"edge_index shape: {edge_index_dummy.shape}\n")

    # ==========================================
    # 2. Instantiate the Model
    # ==========================================
    print("--- Initializing Model ---")
    model = MotionClassifier(
        X=X_dummy, 
        edge_index=edge_index_dummy, 
        num_classes=num_classes, 
        hidden_dim=hidden_dim
    )
    print("Model initialized successfully.\n")

    # ==========================================
    # 3. Test Forward Pass
    # ==========================================
    print("--- Testing Forward Pass ---")
    logits = model(X_dummy, edge_index_dummy)
    print(f"Logits shape: {logits.shape} | Expected: ({B}, {num_classes})")
    assert logits.shape == (B, num_classes), "Output shape mismatch!"
    print("Forward pass successful.\n")

    # ==========================================
    # 4. Test Differentiability (Backward Pass)
    # ==========================================
    print("--- Testing Differentiability ---")
    # Create dummy target labels
    targets = torch.randint(0, num_classes, (B,))
    
    # Calculate loss
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(logits, targets)
    print(f"Initial Loss: {loss.item():.4f}")
    
    # Backpropagate
    loss.backward()
    
    # Check if gradients reached the final classifier
    grad_classifier = model.classifier[0].weight.grad is not None
    
    # Check if gradients reached the diffusion ODE func's sheaf learner
    grad_sheaf = model.sheaf_diffusion.diffuse.sheaf_learner.linear1.weight.grad is not None
    
    # Check if gradients reached the very first linear layer
    grad_lin1 = model.sheaf_diffusion.lin1.weight.grad is not None

    print(f"Gradients at Classifier: {'[OK]' if grad_classifier else '[FAILED]'}")
    print(f"Gradients at Sheaf Learner: {'[OK]' if grad_sheaf else '[FAILED]'}")
    print(f"Gradients at First Layer: {'[OK]' if grad_lin1 else '[FAILED]'}")
    
    assert grad_classifier and grad_sheaf and grad_lin1, "Gradients are broken and did not flow back"
    print("\nSUCCESS: Model is fully differentiable end-to-end")

    # ==========================================
    # 5. Test Diffusion Dynamics
    # ==========================================
    print("--- Testing Diffusion Dynamics ---")
    
    # We will use PyTorch hooks to capture the features right before and after the ODE block
    diffusion_states = {}

    def get_step_io(name):
        def hook(model, input, output):
            # input is a tuple: (t, X)
            diffusion_states[f"{name}_in"] = input[1].detach().clone()
            diffusion_states[f"{name}_out"] = output.detach().clone()
        return hook

    # Register the hook on the discrete diffusion step
    hook_handle = model.sheaf_diffusion.diffuse.register_forward_hook(get_step_io("diffusion_block"))

    # Run a fresh forward pass (no gradients needed for this test)
    with torch.no_grad():
        _ = model(X_dummy, edge_index_dummy)

    # Remove the hook so it doesn't slow down future training
    hook_handle.remove()

    X_before = diffusion_states["diffusion_block_in"]
    X_after = diffusion_states["diffusion_block_out"]

    print(f"Features before diffusion: {X_before.shape}")
    print(f"Features after diffusion:  {X_after.shape}")

    # 1. Check if features actually changed
    feature_diff = torch.norm(X_before - X_after)
    print(f"Total magnitude of feature change: {feature_diff.item():.4f}")
    assert feature_diff > 1e-6, "Features did not change! Diffusion is not doing anything."

    # 2. Check if a completely disconnected graph diffuses differently
    # Let's create an edge_index with NO edges (empty graph)
    empty_edge_index = torch.empty((B, num_confs, 2, 0), dtype=torch.long)
    
    hook_handle = model.sheaf_diffusion.diffuse.register_forward_hook(get_step_io("empty_graph"))
    with torch.no_grad():
        _ = model(X_dummy, empty_edge_index)
    hook_handle.remove()
    
    X_after_empty = diffusion_states["empty_graph_out"]
    
    # If the graph has no edges, the Laplacian should just be block diagonal (self-loops).
    # The diffused features SHOULD be different from the fully connected graph.
    graph_impact = torch.norm(X_after - X_after_empty)
    print(f"Impact of graph structure (Edges vs No Edges): {graph_impact.item():.4f}")
    assert graph_impact > 1e-6, "Graph structure is being ignored! Edges are not routing information."

    print("SUCCESS: Diffusion is actively mixing features based on graph topology!\n")

    # ==========================================
    # 6. Test Global Section Convergence
    # ==========================================
    print("--- Testing Global Section Convergence (Dirichlet Energy) ---")
    
    # 1. RUN A FRESH PASS WITH VALID EDGES to overwrite the empty Laplacian
    with torch.no_grad():
        _ = model(X_dummy, edge_index_dummy)
    
    # 2. Now extract the valid Laplacian
    L_idx, L_val = model.sheaf_diffusion.diffuse.L
    N_total = X_before.size(0)

    def compute_disagreement(X_state):
        # The disagreement is L * X. If X is a perfect global section, L * X = 0.
        LX = torch_sparse.spmm(L_idx, L_val, N_total, N_total, X_state)
        # Return the Frobenius norm (total magnitude) of the disagreement
        return torch.norm(LX).item()

    disagreement_before = compute_disagreement(X_before)
    disagreement_after = compute_disagreement(X_after)

    print(f"Disagreement (||LX||) BEFORE diffusion: {disagreement_before:.4f}")
    print(f"Disagreement (||LX||) AFTER diffusion:  {disagreement_after:.4f}")

    assert disagreement_after < disagreement_before, \
        "Disagreement increased! Diffusion is diverging, likely because `t` is too large."
        
    print("SUCCESS: Features are actively converging toward a global section!\n")
        

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


class DiscreteDiffusionStep(torch.nn.Module):
    """Implements discrete Laplacian-based diffusion step."""

    def __init__(self, d, hidden_channels, 
                 left_weights=False, right_weights=False, 
                 use_act=False, nonlinear=False):
        super(DiscreteDiffusionStep, self).__init__()
        self.d = d
        self.hidden_channels = hidden_channels
        
        # hidden_channels * d is the total node feature size
        self.sheaf_learner = SheafLearner((hidden_channels * d) * 2, (d, d))
        self.nonlinear = nonlinear
        self.left_weights = left_weights
        self.right_weights = right_weights
        self.use_act = use_act
        self.L = None
        
        if self.left_weights:
            self.lin_left_weights = torch.nn.Linear(self.d, self.d, bias=False)
        if self.right_weights:
            self.lin_right_weights = torch.nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)

    def forward(self, dt, X, edge_index):
        # X shape: (num_total_nodes * d, hidden_channels)
        num_total_nodes = X.shape[0] // self.d
        
        if self.nonlinear or self.L is None:
            # Reshape back to (num_total_nodes, hidden_channels * d) for sheaf learning
            X_maps = X.view(num_total_nodes, -1)
            maps = self.sheaf_learner(X_maps, edge_index)
            # Assuming lap.build_norm_sheaf_laplacian returns a tuple of (edge_index, edge_weights)
            self.L = lap.build_norm_sheaf_laplacian(num_total_nodes, self.d, edge_index, maps)
            L = self.L
        else:
            L = self.L
        
        if self.left_weights:
            X = X.t().reshape(-1, self.d)
            X = self.lin_left_weights(X)
            X = X.reshape(-1, num_total_nodes * self.d).t()

        if self.right_weights:
            X = self.lin_right_weights(X)
        
        # Apply sparse matrix multiplication: -L * X
        dX = torch_sparse.spmm(L[0], L[1], X.size(0), X.size(0), -X)

        X = X + (dt * dX)

        if self.use_act:
            X = F.elu(X)

        return X


class SheafDiffusion(torch.nn.Module):
    """Performs diffusion on the sheaf laplacian until global section is reached."""
    def __init__(self, args):
        super().__init__()

        assert args['d'] > 1
        self.d = args['d'] 
        
        self.hidden_dim = args['hidden_channels'] * self.d
        self.device = args.get('device', 'cpu')
        
        self.nonlinear = not args['linear']
        self.input_dropout = args['input_dropout']
        self.dropout = args['dropout']
        self.use_act = args['use_act']
        
        self.input_dim = args['input_dim'] 
        self.hidden_channels = args['hidden_channels'] 
        self.output_dim = args['output_dim']
        self.dt = args['step_size']
        self.steps = args['steps']
        
        self.lin1 = torch.nn.Linear(self.input_dim, self.hidden_dim)
        
        self.diffuse = DiscreteDiffusionStep(
            d=self.d,
            hidden_channels=self.hidden_channels,
            left_weights=args['left_weights'],
            right_weights=args['right_weights'],
            use_act=self.use_act,
            nonlinear=self.nonlinear
        )

    def forward(self, X, edge_index):
        # X: flat batched shape = (B * 2 * num_nodes, num_features)
        
        X = F.dropout(X, p=self.input_dropout, training=self.training) 
        X = self.lin1(X)  # Output shape: (..., hidden_dim)
        
        if self.use_act:
            X = F.elu(X)
            
        X = F.dropout(X, p=self.dropout, training=self.training)

        # Perform diffusion
        if self.dt > 0:
            num_total_nodes = X.shape[0]
            X = X.view(num_total_nodes * self.d, self.hidden_channels)

            self.diffuse.edge_index = edge_index
            self.diffuse.L = None  # Reset Laplacian to ensure it's recomputed if needed

            for _ in range(self.steps):

                X = self.diffuse(self.dt, X, edge_index)

                X = torch.clamp(X, min=-1e4, max=1e4) # prevent param explosion for high step size

                self.diffuse.L = self.diffuse.L # Prevent from rebuilding laplacian on each step.
        
            X = X.view(num_total_nodes, -1)

        assert torch.all(torch.isfinite(X))

        return X


class MotionClassifier(torch.nn.Module):  
    def __init__(self, X, edge_index, K=8, num_classes=5, hidden_dim=64, steps=3, step_size=0.2, input_dropout=0.2, dropout=0.2):
        super().__init__()

        self.input_dim = X.shape[-1]
        self.num_nodes = X.shape[-2]
        self.stalk_dimensions = 3 
        self.hidden_dim = hidden_dim
        self.steps = steps
        self.step_size = step_size
        self.input_dropout = input_dropout
        self.dropout = dropout
        self.K = K

        self.sheaf_diffusion = SheafDiffusion(args={
            'd': self.stalk_dimensions,
            'hidden_channels': self.hidden_dim,
            'steps': self.steps, # number of diffusion steps
            'step_size': self.step_size,
            'linear': False,
            'input_dropout': self.input_dropout,
            'dropout': self.dropout,
            'left_weights': True,
            'right_weights': True,
            'use_act': True,
            'input_dim': self.input_dim,
            'output_dim': self.hidden_dim,
            'device': X.device if isinstance(X, torch.Tensor) else 'cpu'
        })
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim * self.stalk_dimensions * 2, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(self.hidden_dim, num_classes)
        )
        
    def forward(self, X, edge_index):
        # X: shape = (B, 2, num_nodes, num_features)
        # edge_index: shape = (B, 2, 2, num_edges)
        B, num_confs, N, F_dim = X.shape 
        _, _, _, E = edge_index.shape
        
        num_graphs = B * num_confs
        
        # 1. Vectorized edge offsetting
        # Reshape to (B*2, 2, E)
        edge_index_reshaped = edge_index.view(num_graphs, 2, E)
        
        # Create offsets for each graph: [0, N, 2N, 3N, ...] 
        # Shape: (B*2, 1, 1) to broadcast across the 2 node-lists and E edges
        offsets = torch.arange(num_graphs, device=X.device).view(-1, 1, 1) * N
        
        # Apply the offsets so the graphs remain disconnected
        batched_edge_index = edge_index_reshaped + offsets
        
        # Reshape to standard PyG sparse format: (2, B * 2 * E)
        # We transpose to (2, B*2, E) then flatten the last two dims
        batched_edge_index = batched_edge_index.transpose(0, 1).reshape(2, -1)

        # 2. Flatten X
        X_flat = X.view(num_graphs * N, F_dim)

        # 3. Pass both X and the flattened edge_index into diffusion
        X_diffused = self.sheaf_diffusion(X_flat, batched_edge_index)
        
        # 4. Restore original structure (B, 2, N, hidden_dim * d)
        X_diffused = X_diffused.view(B, num_confs, N, -1)

        # 5. Graph Pooling & Classification
        graph_emb = X_diffused.mean(dim=2) 
        combined_emb = torch.cat([graph_emb[:, 0, :], graph_emb[:, 1, :]], dim=-1)
        logits = self.classifier(combined_emb)
        
        return logits
        
        


        
        
        
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
        

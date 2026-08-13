import torch
from torch_geometric.data import Batch
from residue_model import NodeSheafAttentionClassifier

def test_full_architecture():
    # 1. Define Dummy Hyperparameters
    device = torch.device('cpu')
    input_size = 3          # Pure (x,y,z) trajectories
    num_classes = 22        # Amino acids + 2
    hidden_dim = 64
    stalk_dim = 16
    num_blocks = 2
    num_heads = 4
    
    print("Initializing model...")
    model = NodeSheafAttentionClassifier(
        input_size=input_size,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        stalk_dim=stalk_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        ablate_sheaves=False # Keep sheaves enabled to test the hardest paths
    ).to(device)
    model.eval() 

    # 2. Construct a Dummy Batch
    print("Constructing dummy PyG Batch...")
    # Simulating a batch of 2 graphs (proteins)
    # Graph 1: 5 nodes, true sequence length of 20
    # Graph 2: 4 nodes, true sequence length of 15
    total_nodes = 9
    max_seq_len = 20
    
    # Pure trajectory pos: (num_nodes, max_seq_len, 3)
    pos = torch.randn(total_nodes, max_seq_len, 3)
    
    # Graph sequence lengths: (batch_size,)
    lengths = torch.tensor([20, 15], dtype=torch.long)
    
    # Node to graph mapping (data.batch): (total_nodes,)
    batch_idx = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    
    # Dummy edges (ensure no connections exist between graph 1 and 2 to mimic real PyG batches)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 5, 6, 7],
        [1, 2, 3, 4, 6, 7, 8]
    ], dtype=torch.long)
    
    # Make edges symmetric
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    # Dummy labels and train/test mask
    y = torch.randint(0, num_classes, (total_nodes,))
    node_mask = torch.randint(0, 2, (total_nodes,), dtype=torch.bool)
    
    # Create the Batch object
    dummy_batch = Batch(
        pos=pos,
        lengths=lengths,
        batch=batch_idx,
        edge_index=edge_index,
        y=y,
        node_mask=node_mask
    ).to(device)

    # 3. Run Forward Pass
    print("Running forward pass...\n")
    with torch.no_grad():
        try:
            out_batch = model(dummy_batch)
            
            print("Forward pass successful!")
            print(f"-> Output shape: {out_batch.x.shape} (Expected: [{total_nodes}, {num_classes}])\n")
            
            # Verify the Masking Logic for the Loss Function works
            print("Testing loss masking logic...")
            pred_mask = out_batch.node_mask.bool()
            
            test_logits = out_batch.x[pred_mask]
            test_targets = out_batch.y[pred_mask]
            
            num_unmasked = pred_mask.sum().item()
            print(f"-> Masked logits shape: {test_logits.shape} (Expected: [{num_unmasked}, {num_classes}])")
            print(f"-> Masked targets shape: {test_targets.shape} (Expected: [{num_unmasked}])")
            
            # Simulate a loss calculation
            crit = torch.nn.CrossEntropyLoss()
            loss = crit(test_logits, test_targets)
            print(f"-> Dummy Loss Value: {loss.item():.4f}")
            
        except Exception as e:
            print(f"Forward pass failed with error:\n{e}")

if __name__ == "__main__":
    test_full_architecture()

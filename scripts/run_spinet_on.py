import torch
from torch import nn    
from huggingface_hub import login, HfApi, hf_hub_url, hf_hub_download, create_repo
from residue_classifier_dataset import ResidueClassifierDataset
from invariant_features_sheaf_model import NodeSheafClassifier
import os
from load_dynamics import BACKBONE_ATOMS
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from Bio.SeqUtils import seq1
from stats_utils import top_k_acc

@torch.no_grad()
def main(h5_name, index_name, args_dict):
    # hyperparameters
    hidden_dim = 128
    stalk_dim = 8
    num_blocks = 6
    ablate_sheaves=args_dict["ablate_sheaves"]
    ablate_edge_features=args_dict["ablate_edge_features"]
    use_attention = not args_dict["ablate_attention"]
    restriction_map_type=args_dict["restriction_map_type"]
    resume_model_name = args_dict["resume"]
    ablate_order = args_dict["ablate_order"]
    epsilon = 7.5

    df = pd.read_csv(index_name)
    test_set = df[df["split"] == "test"]["domain"].to_list()


    # hf auth
    login(token=os.environ["HF_TOKEN"])
    api = HfApi()

    # device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    ds = ResidueClassifierDataset(h5_name, np.random.default_rng(seed=42), fixed_length=2, groups=test_set, step=1, epsilon=epsilon)
    dataloader = DataLoader(ds, batch_size=8, collate_fn = lambda batch:batch[0])

    # get model
    model = NodeSheafClassifier(
            atoms=BACKBONE_ATOMS,
            frame_origin="CA",
            num_classes=20,
            hidden_dim=hidden_dim,
            stalk_dim=stalk_dim,
            num_blocks=num_blocks,
            num_heads=1,
            ablate_sheaves=ablate_sheaves,
            ablate_edge_features=False,
            num_timesteps=100,
            restriction_map_type=restriction_map_type,
            use_attention=False,
            use_masking=False
        ).to(DEVICE)

    weights_path = hf_hub_download("JensLundsgaard/" + resume_model_name, "pytorch_model.bin", local_dir=os.path.abspath("./"))
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    accs_1 = []
    accs_5 = []
    accs_10 = []
    perplexities = []
    sequences = []
    start_idxs = []
    end_idxs = []
    crit = nn.CrossEntropyLoss()
    for batch in dataloader:
        #batch = next(iter(dataloader))

        batch = batch.sort()
        #print(batch.edge_index.shape)

        batch = batch.to(DEVICE)

        prediction = model(batch)
        prediction = prediction.cpu()
        for pred in prediction.to_data_list():    
            accs_1.append(top_k_acc(pred.x, pred.y, 1)) 
            accs_5.append(top_k_acc(pred.x, pred.y, 5)) 
            accs_10.append(top_k_acc(pred.x, pred.y, 10)) 
            perplexities.append(crit(pred.x, pred.y).exp().item())
            top_logits = pred.x.argmax(dim=-1)
            output_string = "".join([seq1(ds.__class__.AMINO_ACIDS[l]) for l in top_logits.tolist()])
            sequences.append(output_string)
            start_idxs.append(pred.index[1] if pred.index.ndim == 1 else pred.index[0,1])
            end_idxs.append(pred.index[2] if pred.index.ndim == 1 else pred.index[0,2])
    a1 = torch.tensor(accs_1)
    a1m = a1.mean().item()
    a1s = a1.std().item()

    a5 = torch.tensor(accs_5)
    a5m = a5.mean().item()
    a5s = a5.std().item()

    a10 = torch.tensor(accs_10)
    a10m = a10.mean().item()
    a10s = a10.std().item()


    p = torch.tensor(perplexities)
    pm = p.mean().item()
    ps = p.std().item()

    print(f"{sum(p.numel() for p in model.parameters() if p.requires_grad)} & ${a1m:.3f} \\pm {a1s:.3f}$ & ${a5m:.3f} \\pm {a5s:.3f}$ & ${a10m:.3f} \\pm {a10s:.3f}$ & ${pm:.3f} \\pm {ps:.3f}$") # & ${rmsdM: .3f} \\pm {rmsdS: .3f}$") 

    sequence_df = pd.DataFrame({"sequence":sequences,"start_idx":start_idxs, "end_idx":end_idxs})
    print(sequence_df)
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        prog='Train sheaf protein node residue classifier',
        description='Trains sheaf node classifier to predict nodes') 
    parser.add_argument("h5_name")
    parser.add_argument("index_name")
    parser.add_argument('--run-name', type=str, default="default_repo")
    parser.add_argument('--ablate-sheaves', action="store_true")
    parser.add_argument('--ablate-edge-features', action="store_true")
    parser.add_argument('--ablate-attention', action="store_true")
    parser.add_argument('--restriction-map-type', type=str, default="arbitrary", choices=['low_rank', 'orthogonal', 'arbitrary'])
    parser.add_argument('--resume', type=str, default="our_model_mdcath")
    parser.add_argument('--cross-val', type=int, default=0)
    parser.add_argument('--cross-test', type=int, default=4)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument("--ablate-order", action="store_true")

    args = parser.parse_args()
    main(args.h5_name, args.index_name, vars(args))
    

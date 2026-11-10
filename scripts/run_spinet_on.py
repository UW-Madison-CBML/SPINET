import torch
from huggingface_hub import login, HfApi, hf_hub_url, hf_hub_download, create_repo
from residue_classifier_dataset import ResidueClassifierDataset
from invariant_features_sheaf_model import NodeSheafClassifier
import os
from load_dynamics import BACKBONE_ATOMS
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from Bio.SeqUtils import seq1


def main(h5_name, args_dict):
    # hyperparameters
    hidden_dim = 128
    stalk_dim = 8
    num_blocks = 4
    ablate_sheaves=args_dict["ablate_sheaves"]
    ablate_edge_features=args_dict["ablate_edge_features"]
    use_attention = not args_dict["ablate_attention"]
    restriction_map_type=args_dict["restriction_map_type"]
    resume_model_name = args_dict["resume"]
    ablate_order = args_dict["ablate_order"]


    # hf auth
    login(token=os.environ["HF_TOKEN"])
    api = HfApi()


    # device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    ds = ResidueClassifierDataset(h5_name, np.random.default_rng(seed=42), fixed_length=128, groups=None, step=32)
    dataloader = DataLoader(ds, batch_size=1, collate_fn = lambda batch:batch[0])
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

    for motion in dataloader:
        #motion = next(iter(dataloader))

        motion = motion.sort()
        print(motion.edge_index.shape)
        motion = motion.to(DEVICE)

        prediction = model(motion)
        prediction = prediction.cpu()
        logits = prediction.x
        print(logits.shape)
        top_logits = logits.argmax(dim=-1)
        assert top_logits.ndim == 1
        output_string = "".join([seq1(ds.__class__.AMINO_ACIDS[l]) for l in top_logits.tolist()])
        print(motion.index)
        print(output_string)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        prog='Train sheaf protein node residue classifier',
        description='Trains sheaf node classifier to predict nodes') 
    parser.add_argument("h5_name")
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
    main(args.h5_name, vars(args))
    

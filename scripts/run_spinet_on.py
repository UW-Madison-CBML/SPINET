import torch
import hf_hub
from residue_classifier_dataset import ResidueClassifierDataset
from invariant_features_sheaf_model import NodeSheafClassifier

def main(h5_name, args_dict):
    # hyperparameters
    learning_rate = args_dict["lr"]
    epochs = args_dict['epochs']
    hidden_dim = 128
    stalk_dim = 8
    num_blocks = 4
    ablate_sheaves=args_dict["ablate_sheaves"]
    ablate_edge_features=args_dict["ablate_edge_features"]
    use_attention = not args_dict["ablate_attention"]
    num_heads = 4 if use_attention else 1
    run_name = args_dict["run_name"]
    restriction_map_type=args_dict["restriction_map_type"]
    resume_model_name = args_dict["resume"]
    ablate_order = args_dict["ablate_order"]

    # 

    # load data
    ds = ResidueClassifierDataset(h5_name) 
    dataloader = DataLoader(ds, batch_size=1)
    motion = next(iter(dataloader))

    # get model




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
    parser.add_argument('--resume', type=str, default="")
    parser.add_argument('--cross-val', type=int, default=0)
    parser.add_argument('--cross-test', type=int, default=4)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument("--ablate-order", action="store_true")

    args = parser.parse_args()
    train_residue_classifier(args.h5_name, vars(args))
    

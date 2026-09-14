import h5py
from scrmsd import evaluate_batch_rmsd, load_esmfold
import pandas as pd
import numpy as np
import torch
from residue_classifier_dataset import ResidueClassifierDataset
from Bio.SeqUtils import seq1
@torch.no_grad
def test_kabsch_rmsd(data_name="atlas", mdcath_temp=None):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    esmfold_model, tokenizer = load_esmfold()

    pdbs = []
    h5_path = ""
    if data_name == "mdcath":
        h5_path = os.path.abspath(f"mdcath_spinet_{mdcath_temp}.h5")
        index = pd.read_csv(os.path.abspath("mdcath_standard_split.csv"))
        mask = index["split"] != "outside"
        pdbs = index[mask]["domain"].to_list() # each domain is the outer most file

    else:
        h5_path = os.path.abspath("atlas_data.h5")
        index = pd.read_csv(os.path.abspath("atlas_cross_val_index.csv"))
        pdbs = index["pdb"].to_list()

    groups = []
    def visit(name, obj):
        if (isinstance(obj, h5py.Group) and all(ds in obj.keys() for ds in ResidueClassifierDataset.REQUIRED_DATASETS)):
            groups.append(name)
            
    with h5py.File(h5_path, "r") as h5_file:
        h5_file.visititems(visit)

        batch_size = 4
        groups = groups[:-1*(len(groups) % batch_size)] # just get rid of the rest doesn't matter
        assert len(groups) % batch_size == 0
        batches = np.reshape(groups, (-1, batch_size))
        for batch in batches:
            batch_tensors = []
            batch_seqs = []
            for group in batch:
                coords = torch.from_numpy(h5_file[group]["coordinates"][:, :128]) # 128 as in train_residue_classifier
                batch_tensors.append(coords)
                seq = "".join([seq1(ResidueClassifierDataset.AMINO_ACIDS.index(ResidueClassifierDataset.RESIDUE_ALIASES.get(res.decode()[:3], res.decode()[:3]))) for res in h5_file[group_name]["residues"][:]])
                batch_seqs.append(seq)

            batch_mask = []
            

            rmsd = evaluate_batch_rmsd()
            print(f"${rmsd.mean().item():.3f} \\pm {rmsd.std().item():.3f}$")
            del batch_tensors




    







if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(prog='test kabsch_rmsd EsmFold on native sequences') 
    parser.add_argument('--data-name', type=str, default="atlas", choices=["atlas", "mdcath"])
    parser.add_argument('--mdcath-temp', type=int, default=-1, choices=[-1,320, 450, 348, 379])
    args = parser.parse_args()
    assert (args.data_name == "atlas") == (args.mdcath_temp == -1), "do not specify a temp if using atlas"
    test_kabsch_rmsd(args.data_name, mdcath_temp=args.mdcath_temp if args.mdcath_temp != -1 else None)

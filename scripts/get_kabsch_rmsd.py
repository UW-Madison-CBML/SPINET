# get_kabsch_rmsd.py
from scrmsd import evaluate_batch_rmsd, load_esmfold
import sys
import os

import argparse
import io
import urllib.request
from pathlib import Path
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from Bio.PDB import PDBParser

from Bio.PDB.MMCIFParser import MMCIFParser

from load_dynamics import BACKBONE_ATOMS

RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.cif"


def download_pdb(pdb_id: str) -> str:
    url = RCSB_URL.format(pdb_id=pdb_id.upper())
    with urllib.request.urlopen(url) as response:
        return response.read().decode("utf-8")


def extract_backbone(pdb_text: str, pdb_id: str, chain:str) -> torch.Tensor:
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(pdb_id, io.StringIO(pdb_text))

    coords = []
    model = next(structure.get_models())
    for chain in model:
        if not chain.has_id(chain):
            for residue in chain:
                if not residue.has_id("CA"):
                    continue
                try:
                    atom_coords = [residue[atom].coord for atom in BACKBONE_ATOMS]
                except KeyError:
                    continue
                coords.append(atom_coords)

    if not coords:
        raise ValueError(f"No complete backbone residues found in {pdb_id}")

    return torch.tensor(coords, dtype=torch.float32)


def main(seq_csv):
    seq_df = pd.read_csv(os.path.abspath(seq_csv))
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    esmfold_model, esmfold_tokenizer = load_esmfold(device=DEVICE)
    batch_size = 64
    rmsds = []

    # may need to process hdf5 group name
    # if it's just a plain thing it should be robust enough too
    seq_df["pdb"] = seq_df["pdb"].str.split("/").apply(lambda x: x[0] if len(x) > 0 else None)


    # if PPPP_C
    if len(seq_df["pdb"].iloc[0]) == 6:
        seq_df["pdb_id"] = seq_df["pdb"].str.slice(0,4)
        seq_df["chain"] = seq_df["pdb"].str.slice(5,6)
    else:
        seq_df["pdb_id"] = seq_df["pdb"].str.slice(0,4)
        seq_df["chain"] = seq_df["pdb"].str.slice(4,5)


    seq_df["pdb"] = seq_df["pdb_id"] + "_" + seq_df["chain"]

    pdb_names = [name for name, group in seq_df.groupby(["pdb_id","chain"])]

    backbone_dict = {pdb_id + "_" + chain : extract_backbone(download_pdb(pdb_id), pdb_id, chain) for pdb_id, chain in tqdm(pdb_names, desc="getting pdbs")} # in case there are duplicates
    for i in tqdm(range(0, len(seq_df), batch_size), desc="running eval"):
        batch_df = seq_df.iloc[i:min(i+batch_size, len(seq_df)-1)]

        backbone_tensors = [backbone_dict[pdb] for pdb in seq_df["pdb"]]

        lengths = torch.tensor([bb_tensor.shape[0] for bb_tensor in backbone_tensors])

        pad_size = lengths.max().item()

        padded_tensors = [F.pad(bb_tensor, (0,0, 0,0, 0,pad_size-bb_tensor.shape[0]), mode="constant", value=0.0) for bb_tensor in backbone_tensors]

        backbone_tensor = torch.stack(padded_tensors, dim=0) # B, num_res_padded, 4, 3
        backbone_tensor = backbone_tensor.reshape(backbone_tensor.shape[0], -1, 3)
        gt_seq_mask = (lengths[:,None] > torch.arange(pad_size)[None,:])[:, :, None].expand(-1, -1, 4)
        gt_seq_mask = gt_seq_mask.reshape(backbone_tensor.shape[0], -1)

        pred_seqs = batch_df["seq"].to_list()
        rmsd = evaluate_batch_rmsd(pred_seqs, backbone_tensor, gt_seq_mask, esmfold_tokenizer, esmfold_model, device=DEVICE)
        rmsds.extend(rmsd.tolist())
    rmsds = np.array(rmsds)
    print(f"${rmsds.mean().item():.3f} \\pm {rmsds.std().item():.3f}$")


if __name__ == "__main__":
    main(sys.argv[1])

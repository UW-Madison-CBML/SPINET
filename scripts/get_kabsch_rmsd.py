
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
from tqdm import tqdm

from Bio.PDB.MMCIFParser import MMCIFParser

from load_dynamics import BACKBONE_ATOMS

import urllib.request
import urllib.error
import re

RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.cif"

def download_pdb(pdb_id: str, timeout: int = 10) -> str:
    max_tries = 3
    backoff = 1
    for _ in range(max_tries):
        try:
            if not isinstance(pdb_id, str) or not re.fullmatch(r"[0-9A-Za-z]{4}", pdb_id):
                raise ValueError(f"Invalid PDB ID '{pdb_id}'. Must be 4 alphanumeric characters.")

            url = RCSB_URL.format(pdb_id=pdb_id.upper())

            try:
                with urllib.request.urlopen(url, timeout=timeout) as response:
                    if response.status != 200:
                        raise urllib.error.HTTPError(url, response.status, "HTTP error", response.headers, None)
                    data = response.read()
            except urllib.error.HTTPError as e:
                raise urllib.error.HTTPError(e.url, e.code, f"Failed to download PDB file: {e.reason}", e.headers, e.fp)
            except urllib.error.URLError as e:
                raise urllib.error.URLError(f"Network error while downloading PDB file: {e.reason}")

            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                raise ValueError("Downloaded file is not valid UTF-8 text.")

            if not text.strip():
                raise ValueError(f"PDB file for ID '{pdb_id}' is empty.")

            return text
        except ValueError:
            sleep(backoff)
            backoff *= 2
    raise ValueError("reached max backoff")

def extract_backbone(pdb_text: str, pdb_chain_id: str) -> torch.Tensor:
    try:
        pdb_id, chain_id = pdb_chain_id.split("_", 1)
    except ValueError:
        raise ValueError(f"Invalid format '{pdb_chain_id}'. Expected 'PPPP_C'.")

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(pdb_id, io.StringIO(pdb_text))

    coords = []
    model = next(structure.get_models())

    if chain_id not in model:
        return torch.empty((0, len(BACKBONE_ATOMS), 3), dtype=torch.float32)

    chain = model[chain_id]

    for residue in chain:
        if residue.id[0] != " ":
            continue

        if not all(atom_name in residue for atom_name in BACKBONE_ATOMS):
            continue

        atom_coords = [residue[atom_name].coord for atom_name in BACKBONE_ATOMS]
        coords.append(atom_coords)

    if not coords:
        return torch.empty((0, len(BACKBONE_ATOMS), 3), dtype=torch.float32)

    return torch.tensor(coords, dtype=torch.float32)


def main(seq_csv):
    seq_df = pd.read_csv(os.path.abspath(seq_csv + ".csv"))
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    esmfold_model, esmfold_tokenizer = load_esmfold(device=DEVICE)
    batch_size = 64
    rmsds = []
    assert seq_df["pdb"].str.match("^[0-9][A-Za-z0-9]{3}_[A-Za-z0-9]$").all(), "pdb column must match PPPP_C exactly"

    pdb_names = [name for name, group in seq_df.groupby("pdb")]

    backbone_dict = {pdb : extract_backbone(download_pdb(pdb[:4]), pdb) for pdb in tqdm(pdb_names, desc="getting pdbs")} # in case there are duplicates
    print([val.numel() for val in backbone_dict.values()])
    backbone_dict = {key: item for key, item in backbone_dict.items() if item.numel() > 0}
    in_dict = seq_df["pdb"].isin(backbone_dict.keys())
    print(seq_df[~in_dict]["pdb"].unique())
    seq_df = seq_df[in_dict]
    os.makedirs("pred_pdbs", exist_ok=True)
    for i in tqdm(range(0, len(seq_df), batch_size), desc="running eval"):
        batch_df = seq_df.iloc[i:min(i+batch_size, len(seq_df)-1)]

        backbone_tensors = [backbone_dict[pdb] for pdb in batch_df["pdb"]]
        save_pdbs = [os.path.abspath(os.path.join("pred_pdbs", f"{pdb}.pdb")) for pdb in batch_df["pdb"]]

        lengths = torch.tensor([bb_tensor.shape[0] for bb_tensor in backbone_tensors])

        pad_size = lengths.max().item()

        padded_tensors = [F.pad(bb_tensor, (0,0, 0,0, 0,pad_size-bb_tensor.shape[0]), mode="constant", value=0.0) for bb_tensor in backbone_tensors]

        backbone_tensor = torch.stack(padded_tensors, dim=0) # B, num_res_padded, 4, 3
        gt_seq_mask = (lengths[:,None] > torch.arange(pad_size)[None,:])

        pred_seqs = batch_df["seq"].to_list()

        rmsd = evaluate_batch_rmsd(pred_seqs, backbone_tensor, gt_seq_mask,esmfold_model, esmfold_tokenizer, device=DEVICE)
        if rmsd.numel() == 0:
            print("no rmsds returned somehow")
        rmsds.extend(rmsd.tolist())
    rmsds = np.array(rmsds)
    print(f"${rmsds.mean().item():.3f} \\pm {rmsds.std().item():.3f}$")


if __name__ == "__main__":
    main(sys.argv[1])

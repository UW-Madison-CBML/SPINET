import numpy as np
import pandas as pd
import h5py
import wandb
import os
import os.path as osp
import torch
from Bio.PDB import PDBParser
from torch_geometric.loader import DataLoader
import gvp.data
from Bio.SeqUtils import seq1
import torch.nn as nn
import gvp.models
from tqdm import tqdm
import sys
sys.path.append("..")
from stats_utils import get_confusion_matrix, top_k_acc
import json
import io

BACKBONE_ATOMS = ["CA", "N", "C", "O"] # this is the GT order of backbone atoms in a coordinates array
RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.cif"
RESIDUE_ALIASES = {"HSD": "HIS", "HSE": "HIS", "HSP": "HIS", "HID": "HIS", "HIE": "HIS", "HIP": "HIS", "ASH": "ASP", "GLH": "GLU","LYN": "LYS", "CYM": "CYS", "CYX": "CYS", "MSE": "MET"}

import torch
import torch.utils.data as data
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.SeqUtils import seq1


def cache_path(pdb_id: str, cache_dir=os.path.join("..","pdbs")) -> str:
    return os.path.join(cache_dir, f"{pdb_id.upper()}.cif")

def read_cached_cif(pdb_id: str, cache_dir=os.path.join("..","pdbs")) -> str:
    path = cache_path(pdb_id, cache_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No cached .cif for '{pdb_id}' at {path}. "
            f"Run download_cifs.py first to populate the cache."
        )
    with open(path, "r") as f:
        return f.read()


def extract_backbone(pdb_text: str, pdb_chain_id: str):
    try:
        pdb_id, chain_id = pdb_chain_id.split("_", 1)
    except ValueError:
        raise ValueError(f"Invalid format '{pdb_chain_id}'. Expected 'PPPP_C'.")

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(pdb_id, io.StringIO(pdb_text))

    coords = []
    model = next(structure.get_models())

    if chain_id not in model:
        raise ValueError("bad chain id")

    chain = model[chain_id]
    seq = ""
    for residue in chain:
        if residue.id[0] != " ":
            continue

        if not all(atom_name in residue for atom_name in BACKBONE_ATOMS):
            continue
        seq += seq1(residue.get_resname()).upper()
        atom_coords = [residue[atom_name].coord for atom_name in BACKBONE_ATOMS]
        coords.append(atom_coords)

    if not coords:
        raise ValueError("bad coords")

    atom_dict = {
        'name': (pdb_chain_id, -1),
        'seq': seq,
        'coords': coords
    }

    return atom_dict


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_val(model, loader, run, epoch, crit, dataset, val_name="val"):
    model.eval()

    val_acc_top_1 = []
    val_acc_top_5 = []
    val_acc_top_10 = []

    val_losses = []

    seqs = []
    idxs = []
    names = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            nodes = (batch.node_s, batch.node_v)
            edges = (batch.edge_s, batch.edge_v)
            logits = model(nodes, batch.edge_index, edges, batch.seq)

            if batch.mask.any().item():
                print("mask is on for val; may cause issue with downstream residue sizes")
            num_proteins_in_batch = batch.batch.max().item() + 1
            for p_idx in range(num_proteins_in_batch):
                protein_mask = (batch.batch == p_idx) & batch.mask

                if not protein_mask.any():
                    continue

                masked_logits = logits[protein_mask]
                masked_seq = batch.seq[protein_mask]
                loss = crit(masked_logits, masked_seq).item()

                val_losses.append(loss)
                val_acc_top_1.append(top_k_acc(masked_logits, masked_seq, 1))
                val_acc_top_5.append(top_k_acc(masked_logits, masked_seq, 5))
                val_acc_top_10.append(top_k_acc(masked_logits, masked_seq, 10))


                name, idx = batch.name[p_idx]
                names.append(name)
                idxs.append(idx)
                seq_preds = masked_logits.argmax(dim=-1).cpu().tolist()
                seqs.append("".join([dataset.num_to_letter[pred] for pred in seq_preds]))

    seq_pred_df = pd.DataFrame({"seq":seqs, "idx":idxs, "name":names})
    run.log({"seq_pred":wandb.Table(dataframe=seq_pred_df)})
    seq_pred_df.to_csv(os.path.join("..", f"seq_pred_df_{epoch}.csv"))

    val_perps = np.exp(val_losses)
    val_ppl_mean, val_ppl_std = np.mean(val_perps), np.std(val_perps)
    val_t1_mean, val_t1_std = np.mean(val_acc_top_1), np.std(val_acc_top_1)
    val_t5_mean, val_t5_std = np.mean(val_acc_top_5), np.std(val_acc_top_5)
    val_t10_mean, val_t10_std = np.mean(val_acc_top_10), np.std(val_acc_top_10)

    print(f"${val_t1_mean} \\pm {val_t1_std} $ & $ {val_t5_mean} \\pm {val_t5_std}$ & $  {val_t10_mean} \\pm {val_t10_std} $ & $ {val_ppl_mean} \\pm {val_ppl_std} $")


def main(index_path, pdbs_archive, use_pdbs=False):

    wandb.login(key=os.getenv("WANDB_KEY"))
    run = wandb.init(
        entity="jenslundsgaard7-uw-madison",
        project="SheafProtein",
        name="gvp",
        config={
            "index_path":index_path
        },
    )
    if(index_path == "atlas_cross_val_index.csv"):
        split_df = pd.read_csv(osp.join("..", index_path))
        val_mask = split_df["cross_val"] == 0 # change to whatever cross val sets you want
        test_mask = split_df["cross_val"] == 4

        #train_indices = split_df[(~val_mask) & (~test_mask)]["random_indices"].to_list()
        #val_indices = split_df[val_mask]["random_indices"].to_list()
        #test_indices = split_df[test_mask]["random_indices"].to_list()

        val_groups = split_df[val_mask]["pdb"].to_list()
        train_groups = split_df[(~val_mask) & (~test_mask)]["pdb"].to_list()
        test_groups = split_df[test_mask]["pdb"].to_list()
    else:
        index = pd.read_csv(osp.join("..",index_path))

        train_groups = [pdb_id[:4] + "_" + pdb_id[4:5] for pdb_id in  index[index["split"] == "train"]["domain"].tolist()]
        test_groups = [pdb_id[:4] + "_" + pdb_id[4:5] for pdb_id in index[index["split"] == "test"]["domain"].tolist()]
        val_groups = [pdb_id[:4] + "_" + pdb_id[4:5] for pdb_id in index[index["split"] == "validation"]["domain"].tolist()]

        #pdb_to_size = {}
        #def visit(name, obj):
        #    if isinstance(obj, h5py.Group) and "coordinates" in obj:
        #        pdb_to_size[name.split("/")[0]] = obj["coordinates"].shape[1]
        #with h5py.File(h5_path, "r") as f:
        #    f.visititems(visit)
        #index["random_indices"] = [pick_frame(pdb_to_size[pdb_id], pdb_id, 0) for pdb_id in index["pdb"].to_list()]



    train_raw = []
    val_raw = []
    test_raw = []

    train_unique_ids = sorted({pdb[:4] for pdb in train_groups})
    train_cif_text = {}
    for pid in tqdm(train_unique_ids, desc="reading cache"):
        try:
            train_cif_text[pid] = read_cached_cif(pid)
        except FileNotFoundError as e:
            print(e)
    for pdb in tqdm(train_groups, desc="parsing"):
        pid = pdb[:4]
        if pid not in train_cif_text:
            continue  # missing from cache, already logged
        try:
            train_raw.append(extract_backbone(train_cif_text[pid], pdb))
        except ValueError as e:
            print(f"Skipping {pdb}: {e}")





    val_unique_ids = sorted({pdb[:4] for pdb in val_groups})
    val_cif_text = {}
    for pid in tqdm(val_unique_ids, desc="reading cache"):
        try:
            val_cif_text[pid] = read_cached_cif(pid)
        except FileNotFoundError as e:
            print(e)
    for pdb in tqdm(val_groups, desc="parsing"):
        pid = pdb[:4]
        if pid not in val_cif_text:
            continue  # missing from cache, already logged
        try:
            val_raw.append(extract_backbone(val_cif_text[pid], pdb))
        except ValueError as e:
            print(f"Skipping {pdb}: {e}")



    test_unique_ids = sorted({pdb[:4] for pdb in test_groups})
    test_cif_text = {}
    for pid in tqdm(test_unique_ids, desc="reading cache"):
        try:
            test_cif_text[pid] = read_cached_cif(pid)
        except FileNotFoundError as e:
            print(e)
    for pdb in tqdm(test_groups, desc="parsing"):
        pid = pdb[:4]
        if pid not in test_cif_text:
            continue  # missing from cache, already logged
        try:
            test_raw.append(extract_backbone(test_cif_text[pid], pdb))
        except ValueError as e:
            print(f"Skipping {pdb}: {e}")

    """

    else:
        train_raw = []
        val_raw = []
        test_raw = []
        def visit(name, obj):
            if isinstance(obj, h5py.Group) and all(ds in obj for ds in ["coordinates","residues"]):
                pdb = name.split("/")[0]
                idx = index[index["pdb"] == pdb].iloc[0]["random_indices"]
                atom_dict = {
                    'name': (name, idx),
                    'seq': "".join([seq1(res.decode()[:3]) for res in obj["residues"][:]]),
                    'coords': obj["coordinates"][:, idx]
                }
                if(pdb in train_pdbs):
                    train_raw.append(atom_dict)
                else:
                    val_raw.append(atom_dict)
        with h5py.File(h5_path, "r") as f:
            f.visititems(visit)
    """

    train_node_counts = [len(s['seq']) for s in train_raw]
    val_node_counts = [len(s['seq']) for s in val_raw]
    test_node_counts = [len(s['seq']) for s in test_raw]

    train_sampler = gvp.data.BatchSampler(train_node_counts, max_nodes=3000)
    val_sampler = gvp.data.BatchSampler(val_node_counts, max_nodes=3000)
    test_sampler = gvp.data.BatchSampler(test_node_counts, max_nodes=3000)

    train_dataset = gvp.data.ProteinGraphDataset(train_raw)
    val_dataset = gvp.data.ProteinGraphDataset(val_raw)
    test_dataset = gvp.data.ProteinGraphDataset(test_raw)

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=16)
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, num_workers=16)

    model = gvp.models.CPDModel(
        node_in_dim=(6, 3), node_h_dim=(100, 16),
        edge_in_dim=(32, 1), edge_h_dim=(32, 1)
    ).to(DEVICE)

    # Credit: Tomerikoo and Fabio Perez on StackOverflow
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    epochs = 100
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} Train"):
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            nodes = (batch.node_s, batch.node_v)
            edges = (batch.edge_s, batch.edge_v)
            logits = model(nodes, batch.edge_index, edges, batch.seq)

            masked_logits = logits[batch.mask]
            masked_seq = batch.seq[batch.mask]
            loss = crit(masked_logits, masked_seq)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()


        run_val(model, val_loader, run, epoch, crit, val_dataset)


    run_val(model, test_loader, run, epoch, crit, test_dataset, val_name="test")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], False) #  don't use pdbs


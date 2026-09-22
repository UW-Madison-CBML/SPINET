import numpy as np
import pandas as pd
import h5py
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

BACKBONE_ATOMS = ["CA", "N", "C", "O"] # this is the GT order of backbone atoms in a coordinates array
RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.cif"
import zlib
def pick_frame(num_frames, protein_id, seed, frame_index=None):
    if frame_index is not None:
        if not -num_frames <= frame_index < num_frames:
            raise IndexError("frame_index {} out of range for {} ({} frames)".format(
                frame_index, protein_id, num_frames))
        return int(frame_index % num_frames)
    rng = np.random.default_rng([int(seed), zlib.crc32(protein_id.encode())])
    return int(rng.integers(num_frames))

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
    seq = ""
    for residue in chain:
        if residue.id[0] != " ":
            continue

        if not all(atom_name in residue for atom_name in BACKBONE_ATOMS):
            continue
        seq += seq1(residue.get_resname())
        atom_coords = [residue[atom_name].coord for atom_name in BACKBONE_ATOMS]
        coords.append(atom_coords)

    if not coords:
        return torch.empty((0, len(BACKBONE_ATOMS), 3), dtype=torch.float32)

    return torch.tensor(coords, dtype=torch.float32), seq
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_val(model, ):
    model.eval()

    val_acc_top_1 = []
    val_acc_top_5 = []
    val_acc_top_10 = []

    val_losses = []

    seqs = []
    idxs = []
    names = []
    with torch.no_grad():
        for batch in val_loader:
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
                seqs.append("".join([val_dataset.num_to_letter[pred] for pred in seq_preds]))

    seq_pred_df = pd.DataFrame({"seq":seqs, "idx":idxs, "name":names})
    seq_pred_df.to_csv(os.path.join("..", f"seq_pred_df_{epoch}.csv"))
       
    val_perps = np.exp(val_losses)   
    val_ppl_mean, val_ppl_std = np.mean(val_perps), np.std(val_perps)
    val_t1_mean, val_t1_std = np.mean(val_acc_top_1), np.std(val_acc_top_1)
    val_t5_mean, val_t5_std = np.mean(val_acc_top_5), np.std(val_acc_top_5)
    val_t10_mean, val_t10_std = np.mean(val_acc_top_10), np.std(val_acc_top_10)
    
    print(f"${val_t1_mean} \\pm {val_t1_std} $ & $ {val_t5_mean} \\pm {val_t5_std}$ & $  {val_t10_mean} \\pm {val_t10_std} $ & $ {val_ppl_mean} \\pm {val_ppl_std} $")
   

def main(use_pdbs=False, majority_voting=False):
    use_atlas = True
    h5_path = "atlas_data.h5"
    use_pdbs = True
    h5_path = osp.join("..",h5_path)
    temp = 320

    wandb.login(key=os.getenv("WANDB_KEY"))
    run = wandb.init(
        entity="jenslundsgaard7-uw-madison",
        project="SheafProtein",
        name="gvp",
        config={
            "use_atlas":use_atlas
        },
    )
    if(h5_path == "atlas_data.h5"):
        split_df = pd.read_csv(osp.join("..", "atlas_cross_val_index.csv"))
        val_mask = split_df["cross_val"] == 0 # change to whatever cross val sets you want
        test_mask = split_df["cross_val"] == 4

        train_indices = split_df[(~val_mask) & (~test_mask)]["random_indices"].to_list()
        val_indices = split_df[val_mask]["random_indices"].to_list()
        test_indices = split_df[test_mask]["random_indices"].to_list()

        val_groups = split_df[val_mask]["pdb"].to_list()
        train_groups = split_df[(~val_mask) & (~test_mask)]["pdb"].to_list()
        test_groups = split_df[test_mask]["pdb"].to_list()
    else:
        h5_path = osp.join("..", f"mdcath_spinet_{temp}_0.h5")
        index = pd.read_csv(osp.join("..",f"mdcath_{temp}_0_topology_split.csv"))
        index = index.rename(columns={"domain":"pdb"})

        train_groups = index[index["split"] == "train"]["pdb"].tolist()
        test_groups = index[index["split"] == "test"]["pdb"].tolist()
        val_groups = index[index["split"] == "validation"]["pdb"].tolist()

        pdb_to_size = {}
        def visit(name, obj):
            if isinstance(obj, h5py.Group) and "coordinates" in obj:
                pdb_to_size[name.split("/")[0]] = obj["coordinates"].shape[1]
        with h5py.File(h5_path, "r") as f:
            f.visititems(visit)
        index["random_indices"] = [pick_frame(pdb_to_size[pdb_id], pdb_id, 0) for pdb_id in index["pdb"].to_list()]


    
    if use_pdbs:
        # TODO manualy download RCSB structures, and use the above func to train on them
        train_raw = [] 
        val_raw = [] 

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
    
    epochs = 8
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

       
   




if __name__ == '__main__':
    main(False) #  don't use pdbs


import numpy as np
import pandas as pd
import h5py
import os
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

def parse_pdb_folder(folder_path):
    parser = PDBParser(QUIET=True)
    parsed_structures = []
    for file_name in os.listdir(folder_path):
        if not file_name.endswith('.pdb'):
            continue
        file_path = os.path.join(folder_path, file_name)
        structure_id = os.path.splitext(file_name)[0]
        structure = parser.get_structure(structure_id, file_path)
        
        for model in structure:
            for chain in model:
                seq_chars = []
                coords_list = []
                for residue in chain:
                    res_name = residue.get_resname()
                    seq_chars.append(seq1(res_name))
                    res_coords = [residue[atom].get_coord() for atom in BACKBONE_ATOMS]
                    coords_list.append(res_coords)
                if len(seq_chars) > 0:
                    coords_np = np.array(coords_list, dtype=np.float32)
                    parsed_structures.append({
                        'name': structure_id,
                        'seq': "".join(seq_chars),
                        'coords': coords_np
                    })
                break
            break
    return parsed_structures

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(use_pdbs=False, majority_voting=False):
    h5_path = os.path.join("..", "atlas_data.h5")
    index = pd.read_csv(os.path.join("..", "atlas_cross_val_index.csv"))

    val_mask = index["cross_val"] == 0

    val_pdbs = index[val_mask]["pdb"].to_list()
    train_pdbs = index[~val_mask]["pdb"].to_list()


    if use_pdbs:
        # TODO manualy download RCSB structures, and use the above func to train on them
        train_raw = [] #parse_pdb_folder("./surffold_data/train")
        val_raw = [] #parse_pdb_folder("./surffold_data/validation")

    else:
        train_raw = []
        val_raw = []
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
    
    train_sampler = gvp.data.BatchSampler(train_node_counts, max_nodes=3000)
    val_sampler = gvp.data.BatchSampler(val_node_counts, max_nodes=3000)
    
    train_dataset = gvp.data.ProteinGraphDataset(train_raw)
    val_dataset = gvp.data.ProteinGraphDataset(val_raw)
    
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=16)
    
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
                
                if batch.mask.any():
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
                    seqs.append("".join([val_dataset.num_to_letter(pred) for pred in seq_preds]))

        seq_pred_df = pd.DataFrame({"seq":seqs, "idx":idxs, "name":names})
        seq_pred_df.to_csv(os.path.join("..", f"seq_pred_df_{epoch}.csv"))
           
        val_perps = np.exp(val_losses)   
        val_ppl_mean, val_ppl_std = np.mean(val_perps), np.std(val_perps)
        val_t1_mean, val_t1_std = np.mean(val_acc_top_1), np.std(val_acc_top_1)
        val_t5_mean, val_t5_std = np.mean(val_acc_top_5), np.std(val_acc_top_5)
        val_t10_mean, val_t10_std = np.mean(val_acc_top_10), np.std(val_acc_top_10)
        
        print(f"Val Perplexity: {val_ppl_mean} \pm {val_ppl_std}")
        print(f"Top-1 Recovery: {val_t1_mean} \pm {val_t1_std}")
        print(f"Top-5 Recovery: {val_t5_mean} \pm {val_t5_std}")
        print(f"Top-10 Recovery: {val_t10_mean} \pm {val_t10_std}")
        


if __name__ == '__main__':
    main(False) #  don't use pdbs


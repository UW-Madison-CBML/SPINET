import numpy as np
import os
import torch
from Bio.PDB import PDBParser
from torch_geometric.loader import DataLoader
import gvp.data  
from Bio.SeqUtils import seq1
import torch.nn as nn
import gvp.models
from tqdm import tqdm
sys.path.append("..")
from stats_utils import get_confusion_mat, top_k_acc

AA_MAP = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
    'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V'
}

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
                    backbone_atoms = ['N', 'CA', 'C', 'O']
                    has_all_atoms = all(atom in residue for atom in backbone_atoms)
                    if res_name in AA_MAP and has_all_atoms:
                        seq_chars.append(AA_MAP[res_name])
                        res_coords = [residue[atom].get_coord() for atom in backbone_atoms]
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

DEVICE = torch.DEVICE('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    train_raw = parse_pdb_folder("./surffold_data/train")
    val_raw = parse_pdb_folder("./surffold_data/validation")
    
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
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    
    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} Train"):
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            nodes = (batch.node_s, batch.node_v)
            edges = (batch.edge_s, batch.edge_v)
            preds = model(nodes, batch.edge_index, edges, batch.seq)
           
            masked_preds = preds[batch.mask]
            masked_seq = batch.seq[batch.mask]
            loss = crit(masked_preds, masked_seq)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()

        global_confusion_mat = torch.zeros((num_classes, num_classes))
        val_acc_top_1 = []
        val_acc_top_5 = []
        val_acc_top_10 = []

        pos_correct = torch.zeros(0, dtype=torch.long)
        pos_total = torch.zeros(0, dtype=torch.long)

        val_losses = []

        f1s = {acid:[] for acid in ResidueClassifierDataset.AMINO_ACIDS}
        precisions = {acid:[] for acid in ResidueClassifierDataset.AMINO_ACIDS}
        recalls = {acid:[] for acid in ResidueClassifierDataset.AMINO_ACIDS}
 
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                nodes = (batch.node_s, batch.node_v)
                edges = (batch.edge_s, batch.edge_v)
                preds = model(nodes, batch.edge_index, edges, batch.seq)
                
                raw_loss = loss_fn(preds, batch.seq)
                
                _, topk_indices = torch.topk(preds, k=10, dim=-1)
                targets_expanded = batch.seq.unsqueeze(-1)
                
                match_top1 = (topk_indices[:, :1] == targets_expanded).any(dim=-1).float()
                match_top5 = (topk_indices[:, :5] == targets_expanded).any(dim=-1).float()
                match_top10 = (topk_indices[:, :10] == targets_expanded).any(dim=-1).float()
                
                num_proteins_in_batch = batch.batch.max().item() + 1
                prot_perplexities = []
                prot_rec1 = []
                prot_rec5 = []
                prot_rec10 = []
                
                for p_idx in range(num_proteins_in_batch):
                    protein_mask = (batch.batch == p_idx) & batch.mask
                    if not protein_mask.any():
                        continue
                        
                    p_loss = raw_loss[protein_mask].mean().item()
                    prot_perplexities.append(np.exp(p_loss))
                    
                    prot_rec1.append(match_top1[protein_mask].mean().item())
                    prot_rec5.append(match_top5[protein_mask].mean().item())
                    prot_rec10.append(match_top10[protein_mask].mean().item())
                
                if prot_perplexities:
                    batch_perplexities.append(np.mean(prot_perplexities))
                    batch_top1.append(np.mean(prot_rec1))
                    batch_top5.append(np.mean(prot_rec5))
                    batch_top10.append(np.mean(prot_rec10))
                    
        val_ppl_mean, val_ppl_std = np.mean(batch_perplexities), np.std(batch_perplexities)
        val_t1_mean, val_t1_std = np.mean(batch_top1), np.std(batch_top1)
        val_t5_mean, val_t5_std = np.mean(batch_top5), np.std(batch_top5)
        val_t10_mean, val_t10_std = np.mean(batch_top10), np.std(batch_top10)
        
        print(f"Train Loss    : {total_loss/len(train_loader)}")
        print(f"Val Perplexity: {val_ppl_mean} \pm {val_ppl_std}")
        print(f"Top-1 Recovery: {val_t1_mean} \pm {val_t1_std}")
        print(f"Top-5 Recovery: {val_t5_mean} \pm {val_t5_std}")
        print(f"Top-10 Recovery: {val_t10_mean} \pm {val_t10_std}")

if __name__ == '__main__':
    main()


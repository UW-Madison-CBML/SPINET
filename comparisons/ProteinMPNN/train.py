
"""
Modified from gvp-pytorch/train.py (author: Jens Lundsgaard) via Claude to match ProteinMPNN's formatting
"""

import os
import os.path as osp
import sys

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from Bio.SeqUtils import seq1
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm

import gvp.data  # only kept for gvp.data.BatchSampler (token-budget train batching)

# ProteinMPNN: clone https://github.com/dauparas/ProteinMPNN and put ProteinMPNN/training on PYTHONPATH.
# Use the *training* copy: it has the 6-arg forward (random decoding order drawn internally)
# and get_std_opt. The root protein_mpnn_utils.py forward needs an extra `randn` arg.
from model_utils import ProteinMPNN, get_std_opt

letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8, 'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19,'N': 2, 'Y': 18, 'M': 12}
num_to_letter = {v:k for k, v in letter_to_num.items()}
BACKBONE_ATOMS = ["CA", "N", "C", "O"] # this is the GT order of backbone atoms in a coordinates array
RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.cif"
RESIDUE_ALIASES = {"HSD": "HIS", "HSE": "HIS", "HSP": "HIS", "HID": "HIS", "HIE": "HIS", "HIP": "HIS", "ASH": "ASP", "GLH": "GLU","LYN": "LYS", "CYM": "CYS", "CYX": "CYS", "MSE": "MET"}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from stats_utils import get_confusion_matrix, top_k_acc


MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"  # ProteinMPNN's order; X=20 is "unknown"
ALPHA_IDX = {a: i for i, a in enumerate(MPNN_ALPHABET)}


class MPNNDataset(data.Dataset):
    """Replaces gvp.data.ProteinGraphDataset. No graph building: ProteinMPNN does kNN internally.
    Exposes letter_to_num / num_to_letter over the 20 real amino acids so run_val's metrics
    code works unchanged (X is excluded from per-AA metrics)."""
    letter_to_num = {a: i for i, a in enumerate(MPNN_ALPHABET[:20])}
    num_to_letter = {i: a for a, i in letter_to_num.items()}

    def __init__(self, raw):
        self.raw = raw

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, i):
        d = self.raw[i]
        return d["name"], d["seq"], d["coords"]


def mpnn_collate(items):
    """Pad to a dense batch. coords must be (L, 4, 3) in N, CA, C, O order (same as GVP's input)."""
    B = len(items)
    L = max(len(seq) for _, seq, _ in items)
    X = torch.zeros(B, L, 4, 3)
    S = torch.zeros(B, L, dtype=torch.long)
    mask = torch.zeros(B, L)
    for b, (_, seq, coords) in enumerate(items):
        n = len(seq)
        c = torch.as_tensor(np.asarray(coords), dtype=torch.float32)
        mask[b, :n] = torch.isfinite(c).all(dim=-1).all(dim=-1).float()  # same role as GVP's batch.mask
        X[b, :n] = torch.nan_to_num(c, nan=0.0)
        S[b, :n] = torch.tensor([ALPHA_IDX.get(a, 20) for a in seq])
    return {
        "name": [name for name, _, _ in items],
        "X": X,
        "S": S,
        "mask": mask,
        "chain_M": mask.clone(),                                   # predict every valid residue (single chain)
        "residue_idx": torch.arange(L).unsqueeze(0).repeat(B, 1),  # contiguous numbering
        "chain_enc": torch.ones(B, L),                             # one chain
    }


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def mpnn_forward(model, b):
    """Returns log_probs (B, L, 21) -- already log-softmaxed, so use NLLLoss, NOT CrossEntropyLoss."""
    return model(b["X"], b["S"], b["mask"], b["chain_M"], b["residue_idx"], b["chain_enc"])


def run_val(model, loader, run, epoch, crit, dataset, val_name="val"):
    model.eval()

    val_acc_top_1 = []
    val_acc_top_5 = []
    val_acc_top_10 = []

    val_losses = []
    num_classes = len(dataset.num_to_letter.keys())  # 20
    global_confusion_mat = torch.zeros((num_classes, num_classes))
    f1s = {acid: [] for acid in dataset.letter_to_num.keys()}
    precisions = {acid: [] for acid in dataset.letter_to_num.keys()}
    recalls = {acid: [] for acid in dataset.letter_to_num.keys()}
    seqs = []
    idxs = []
    names = []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, DEVICE)
            log_probs = mpnn_forward(model, batch)  # (num_frames, L, 21)

            # average over the trajectory window. mean of log-probs then log_softmax == softmax of the
            # mean logits, i.e. identical to your GVP logit averaging. Slice to the 20 real AAs
            # (drop X) so metrics/top-k are over 20 classes.
            logits_avg = torch.log_softmax(log_probs.mean(dim=0)[:, :20], dim=-1)  # (L, 20)
            seq = batch["S"][0]  # same sequence in every frame
            keep = batch["mask"].bool().all(dim=0)  # drop residues with non-finite coords in any frame
            logits_avg, seq = logits_avg[keep], seq[keep]

            loss = crit(logits_avg, seq).item()
            batch_conf_mat = get_confusion_matrix(seq.cpu(), logits_avg.argmax(dim=-1).cpu(), num_classes)
            global_confusion_mat += batch_conf_mat

            diag = batch_conf_mat.diag()
            recall = torch.nan_to_num(diag / batch_conf_mat.sum(dim=1), 0.0)
            precision = torch.nan_to_num(diag / batch_conf_mat.sum(dim=0), 0.0)
            f1 = torch.nan_to_num(2 * (precision * recall) / (precision + recall), 0.0)
            for k, amino_acid in enumerate(dataset.letter_to_num.keys()):
                precisions[amino_acid].append(precision[k].item())
                recalls[amino_acid].append(recall[k].item())
                f1s[amino_acid].append(f1[k].item())

            val_losses.append(loss)
            val_acc_top_1.append(top_k_acc(logits_avg, seq, 1))
            val_acc_top_5.append(top_k_acc(logits_avg, seq, 5))
            val_acc_top_10.append(top_k_acc(logits_avg, seq, 10))

            name, idx = batch["name"][-1]
            names.append(name)
            idxs.append(idx)
            seq_preds = logits_avg.argmax(dim=-1).cpu().tolist()
            seqs.append("".join([dataset.num_to_letter[pred] for pred in seq_preds]))

    prf_dict = {}
    precisions = {key: torch.tensor(value) for key, value in precisions.items()}
    recalls = {key: torch.tensor(value) for key, value in recalls.items()}
    f1s = {key: torch.tensor(value) for key, value in f1s.items()}
    for k, amino_acid in enumerate(dataset.letter_to_num.keys()):
        prf_dict[f"{val_name}_{amino_acid}_f1_mean"] = f1s[amino_acid].mean().item()
        prf_dict[f"{val_name}_{amino_acid}_precision_mean"] = precisions[amino_acid].mean().item()
        prf_dict[f"{val_name}_{amino_acid}_recall_mean"] = recalls[amino_acid].mean().item()
        prf_dict[f"{val_name}_{amino_acid}_f1_std"] = f1s[amino_acid].std().item()
        prf_dict[f"{val_name}_{amino_acid}_precision_std"] = precisions[amino_acid].std().item()
        prf_dict[f"{val_name}_{amino_acid}_recall_std"] = recalls[amino_acid].std().item()
    seq_pred_df = pd.DataFrame({"seq": seqs, "idx": idxs, "name": names})
    run.log({"seq_pred": wandb.Table(dataframe=seq_pred_df)} | prf_dict)
    # FIX: include val_name, otherwise the final test call overwrites the last val epoch's CSV
    seq_pred_df.to_csv(os.path.join("..", f"seq_pred_df_{val_name}_{epoch}.csv"))

    val_perps = np.exp(val_losses)
    val_ppl_mean, val_ppl_std = np.mean(val_perps), np.std(val_perps)
    val_t1_mean, val_t1_std = np.mean(val_acc_top_1), np.std(val_acc_top_1)
    val_t5_mean, val_t5_std = np.mean(val_acc_top_5), np.std(val_acc_top_5)
    val_t10_mean, val_t10_std = np.mean(val_acc_top_10), np.std(val_acc_top_10)

    print(f"${val_t1_mean:.3f} \\pm {val_t1_std:.3f} $ & $ {val_t5_mean:.3f} \\pm {val_t5_std:.3f}$ & $  {val_t10_mean:.3f} \\pm {val_t10_std:.3f} $ & $ {val_ppl_mean:.3f} \\pm {val_ppl_std:.3f} $")


class AvgLogitsBatchSampler(data.BatchSampler):
    """
    return a bunch of frames from the same trajectory together
    """
    def __init__(self, raw_data, subsequence=128, step=32):
        super().__init__([], 1, False)
        list_names = [d["name"][0] for d in raw_data]
        self.names = list(dict.fromkeys(list_names))  # FIX: set() order changes per process (hash randomization)
        order = {n: i for i, n in enumerate(self.names)}
        self.index = sorted(enumerate(list_names), key=lambda tup: order[tup[1]])  # stable: keeps frame order
        batches = []
        save_idx = 0
        for name in self.names:
            batch = []
            while save_idx < len(self.index) and self.index[save_idx][1] == name:
                batch.append(self.index[save_idx][0])
                save_idx += 1
            if batch:
                batches.append(batch)
        self.subsequence = subsequence
        self.step = step
        # FIX: `for range(...)` was a SyntaxError (missing `i in`)
        self.batches = [[batch[i:i + self.subsequence] for i in range(0, len(batch) - self.subsequence + 1, self.step)]
                        for batch in batches]
        # FIX: flatten self.batches (you flattened `batches`, which yields ints, not index lists)
        self.batches = [b for batch in self.batches for b in batch]

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def main(index_path, h5_path, use_pdbs=False):
    step = 16

    wandb.login(key=os.getenv("WANDB_KEY"))
    run = wandb.init(
        entity="jenslundsgaard7-uw-madison",
        project="SheafProtein",
        name="proteinmpnn",
        config={
            "index_path": index_path
        },
    )

    h5_path = osp.join("..", h5_path)

    if index_path == "atlas_cross_val_index.csv":
        split_df = pd.read_csv(osp.join("..", index_path))
        val_mask = split_df["cross_val"] == 0  # change to whatever cross val sets you want
        test_mask = split_df["cross_val"] == 4

        val_groups = split_df[val_mask]["pdb"].to_list()
        train_groups = split_df[(~val_mask) & (~test_mask)]["pdb"].to_list()
        test_groups = split_df[test_mask]["pdb"].to_list()
    else:
        index = pd.read_csv(osp.join("..", index_path))

        train_groups = index[index["split"] == "train"]["domain"].tolist()
        test_groups = index[index["split"] == "test"]["domain"].tolist()
        val_groups = index[index["split"] == "validation"]["domain"].tolist()

    # sets: `in` on lists is O(n) per H5 group
    train_set, val_set, test_set = set(train_groups), set(val_groups), set(test_groups)

    train_raw = []
    val_raw = []
    test_raw = []

    def visit(name, obj):
        if isinstance(obj, h5py.Group) and all(ds in obj for ds in ["coordinates", "residues"]):
            pdb = name.split("/")[0].strip()

            # FIX: was `pdb in test_raw` (a list of dicts -> always False), so test was silently empty
            if not (pdb in train_set or pdb in val_set or pdb in test_set):
                return

            seq = "".join([seq1(RESIDUE_ALIASES.get(res.decode()[:3], res.decode()[:3])) for res in obj["residues"][:]])
            coords = obj["coordinates"][:, ::(step if pdb in train_set else 1)]  # we can train on subset and eval on all

            for idx in range(coords.shape[1]):
                atom_dict = {
                    'name': (name, idx * step),
                    'seq': seq,
                    'coords': coords[:, idx]
                }
                if pdb in train_set:
                    train_raw.append(atom_dict)
                elif pdb in val_set:
                    val_raw.append(atom_dict)
                else:
                    test_raw.append(atom_dict)

    with h5py.File(h5_path, "r") as f:
        f.visititems(visit)

    print(f"frames: train={len(train_raw)} val={len(val_raw)} test={len(test_raw)}")
    assert len(test_raw) > 0 and len(val_raw) > 0 and len(train_raw) > 0, "empty split"

    train_node_counts = [len(s['seq']) for s in train_raw]

    train_sampler = gvp.data.BatchSampler(train_node_counts, max_nodes=3000)
    val_sampler = AvgLogitsBatchSampler(val_raw)
    test_sampler = AvgLogitsBatchSampler(test_raw)

    train_dataset = MPNNDataset(train_raw)
    val_dataset = MPNNDataset(val_raw)
    test_dataset = MPNNDataset(test_raw)

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=16, collate_fn=mpnn_collate)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=16, collate_fn=mpnn_collate)
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, num_workers=16, collate_fn=mpnn_collate)

    # ProteinMPNN defaults from training.py: hidden 128, 3+3 layers, 48 neighbors, dropout 0.1, backbone noise 0.2
    model = ProteinMPNN(
        num_letters=21, node_features=128, edge_features=128, hidden_dim=128,
        num_encoder_layers=3, num_decoder_layers=3, vocab=21,
        k_neighbors=48, augment_eps=0.2, dropout=0.1,
    ).to(DEVICE)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)

    optimizer = get_std_opt(model.parameters(), 128, 0)  # Noam schedule (warmup 4000), as in the ProteinMPNN repo
    crit = nn.NLLLoss()  # model outputs log-probs; CrossEntropyLoss would log-softmax twice

    epochs = 20
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} Train"):
            batch = to_device(batch, DEVICE)
            optimizer.zero_grad()
            log_probs = mpnn_forward(model, batch)

            m = batch["mask"].bool()
            loss = crit(log_probs[m], batch["S"][m])

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        run_val(model, val_loader, run, epoch, crit, val_dataset)

    run_val(model, test_loader, run, epoch, crit, test_dataset, val_name="test")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], False)  # don't use pdbs

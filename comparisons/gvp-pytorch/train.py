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
letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8, 'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19,'N': 2, 'Y': 18, 'M': 12}
num_to_letter = {v:k for k, v in letter_to_num.items()}
BACKBONE_ATOMS = ["CA", "N", "C", "O"] # this is the GT order of backbone atoms in a coordinates array
RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.cif"
RESIDUE_ALIASES = {"HSD": "HIS", "HSE": "HIS", "HSP": "HIS", "HID": "HIS", "HIE": "HIS", "HIP": "HIS", "ASH": "ASP", "GLH": "GLU","LYN": "LYS", "CYM": "CYS", "CYX": "CYS", "MSE": "MET"}

import torch
import torch.utils.data as data
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.SeqUtils import seq1
import requests
import os
import time
BACKBONE_ATOMS = ("CA", "N", "C", "O")
import torch
import torch.utils.data as data
from tqdm import tqdm
from Bio.SeqUtils import seq1
import h5py
import numpy as np
import tempfile
from concurrent.futures import ThreadPoolExecutor

RESIDUE_ALIASES = {"HSD": "HIS", "HSE": "HIS", "HSP": "HIS", "HID": "HIS",
                   "HIE": "HIS", "HIP": "HIS", "ASH": "ASP", "GLH": "GLU",
                   "LYN": "LYS", "CYM": "CYS", "CYX": "CYS", "MSE": "MET",
                   "SEC": "CYS", "PYL": "LYS"}

from Bio import Align, PDB
from Bio.Data import IUPACData

_HEADERS = {"User-Agent": "cmikulski@wisc.edu"} # hide!
THREE_TO_ONE = {name.upper(): letter for name, letter in IUPACData.protein_letters_3to1.items()}
RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.{ext}"
def three_to_one(resname):
    resname = RESIDUE_ALIASES.get(resname.strip().upper(), resname.strip().upper())
    return THREE_TO_ONE.get(resname)


def parse_structure_id(protein_id):
    """``protein_id`` -> ``(pdb_id, chain_id)``.

    Understands ATLAS's ``1a0a_A`` / ``1a0a:A`` and mdCATH's CATH domain ids (``1a0aA02``).
    A bare 4-character code yields an empty chain id, which `load_chain_backbone` reads as
    "use the first protein chain".
    """
    protein_id = protein_id.strip().replace(":", "_")
    if "_" in protein_id:
        pdb_id, chain_id = protein_id.split("_", 1)
        return pdb_id[:4].upper(), chain_id.strip()[:1]
    if len(protein_id) >= 5:
        # CATH domain id: 4-char pdb code, then the chain character (the trailing 2 digits
        # are the domain number, which the deposited file knows nothing about).
        return protein_id[:4].upper(), protein_id[4]
    return protein_id[:4].upper(), ""


def fetch_structure_file(pdb_id, cache_dir, attempts=6):
    """Download ``<pdb_id>`` from RCSB into ``cache_dir``, returning the local path.

    Prefers the legacy ``.pdb`` format (what Biopython's PDBParser and DSSP are happiest
    with) and falls back to ``.cif`` for entries too large to be distributed as PDB.
    Already-downloaded files are reused, so re-running featurization is free.
    """
    os.makedirs(cache_dir, exist_ok=True)
    for ext in ("pdb", "cif"):
        cached = os.path.join(cache_dir, "{}.{}".format(pdb_id.upper(), ext))
        if os.path.exists(cached) and os.path.getsize(cached) > 0:
            return cached

    last_error = None
    for ext in ("pdb", "cif"):
        url = RCSB_DOWNLOAD_URL.format(pdb_id=pdb_id.upper(), ext=ext)
        delay = 1.0
        for _ in range(attempts):
            try:
                response = requests.get(url, headers=_HEADERS, timeout=(10, 120))
            except requests.exceptions.RequestException as exc:
                last_error = exc
                time.sleep(delay)
                delay *= 1.5
                continue
            if response.status_code == 200:
                target = os.path.join(cache_dir, "{}.{}".format(pdb_id.upper(), ext))
                # Write via a temp file in the same directory so concurrent downloaders
                # never observe a half-written cache entry.
                fd, tmp_path = tempfile.mkstemp(dir=cache_dir, suffix=".part")
                with os.fdopen(fd, "w") as handle:
                    handle.write(response.text)
                os.replace(tmp_path, target)
                return target
            if response.status_code == 404:
                last_error = "404 for {}".format(url)
                break  # no point retrying a missing format; try the next one
            last_error = "{} for {}".format(response.status_code, url)
            time.sleep(delay)
            delay *= 1.5
    raise ValueError("could not download {} from RCSB: {}".format(pdb_id, last_error))


def _parse_structure(path, pdb_id):
    parser = PDB.MMCIFParser(QUIET=True) if path.endswith(".cif") else PDB.PDBParser(QUIET=True)
    return parser.get_structure(pdb_id, path)


def load_chain_backbone(path, pdb_id, chain_id=""):
    """Parse one chain of a deposited structure into a record dict.

    Returns ``{'pdb_id', 'chain_id', 'seq', 'resnames', 'res_ids', 'coords', 'mask',
    'residues', 'path'}`` where ``coords`` is ``(R, 4, 3)`` in `BACKBONE_ATOMS` order and
    ``mask`` is ``(R,)`` bool, True where all four backbone atoms are present. Only the first
    model, standard amino acids, and non-hetero residues are kept.
    """
    model = _parse_structure(path, pdb_id)[0]

    chain = None
    if chain_id:
        for candidate in (chain_id, chain_id.upper(), chain_id.lower()):
            if candidate in model:
                chain = model[candidate]
                break
    if chain is None:
        # No chain id given, or the requested one is absent (mmCIF label vs auth mismatches
        # do happen) -- fall back to the first chain with any standard amino acids.
        for candidate in model:
            if any(three_to_one(res.get_resname()) for res in candidate):
                chain = candidate
                break
    if chain is None:
        raise ValueError("{}: no protein chain found (wanted chain {!r})".format(pdb_id, chain_id))

    residues, resnames, res_ids, letters = [], [], [], []
    coords, mask = [], []
    for residue in chain:
        if residue.id[0] != " ":
            continue
        letter = three_to_one(residue.get_resname())
        if letter is None:
            continue

        atom_xyz = np.zeros((len(BACKBONE_ATOMS), 3), dtype=np.float32)
        complete = True
        for i, atom_name in enumerate(BACKBONE_ATOMS):
            if atom_name in residue:
                atom_xyz[i] = residue[atom_name].get_coord()
            else:
                complete = False
        if not complete:
            # A residue missing backbone atoms cannot be superposed or written out as a
            # usable PDB record; dropping it keeps `coords`/`seq` in lock-step.
            continue

        residues.append(residue)
        resnames.append(RESIDUE_ALIASES.get(residue.get_resname().upper(), residue.get_resname().upper()))
        res_ids.append(residue.id[1])
        letters.append(letter)
        coords.append(atom_xyz)
        mask.append(True)

    if not residues:
        raise ValueError("{}: chain {} has no complete standard residues".format(pdb_id, chain.id))

    return {
        "pdb_id": pdb_id.upper(),
        "chain_id": chain.id,
        "seq": "".join(letters),
        "resnames": resnames,
        "res_ids": res_ids,
        "coords": np.stack(coords, axis=0),
        "mask": np.array(mask, dtype=bool),
        "residues": residues,
        "path": path,
    }


def _make_aligner():
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    # Terminal gaps are free: the deposited chain routinely carries expression tags, and an
    # mdCATH domain covers only part of it, so both ends legitimately hang off.
    aligner.target_end_gap_score = 0.0
    aligner.query_end_gap_score = 0.0
    return aligner


def align_sequences(query_seq, target_seq):
    """Aligned index pairs between two sequences.

    Returns ``(query_idx, target_idx)``, two equal-length int arrays of positions that the
    global alignment pairs up *and* that carry the same amino acid. Identity-only pairing is
    deliberate: mismatches mean the two sources disagree about that residue, which makes it
    useless as a structural correspondence.
    """
    if not query_seq or not target_seq:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)

    alignment = _make_aligner().align(query_seq, target_seq)[0]
    query_blocks, target_blocks = alignment.aligned[:2]

    query_idx, target_idx = [], []
    for (q_start, q_end), (t_start, _) in zip(query_blocks, target_blocks):
        for offset in range(q_end - q_start):
            q_i, t_i = q_start + offset, t_start + offset
            if query_seq[q_i] == target_seq[t_i]:
                query_idx.append(q_i)
                target_idx.append(t_i)
    return np.array(query_idx, dtype=int), np.array(target_idx, dtype=int)


def subset_record(record, indices):
    """A new record keeping only ``indices`` (an int array of residue positions)."""
    indices = np.asarray(indices, dtype=int)
    subset = dict(record)
    subset["seq"] = "".join(record["seq"][i] for i in indices)
    subset["resnames"] = [record["resnames"][i] for i in indices]
    subset["res_ids"] = [record["res_ids"][i] for i in indices]
    subset["coords"] = record["coords"][indices]
    subset["mask"] = record["mask"][indices]
    subset["residues"] = [record["residues"][i] for i in indices]
    return subset


def crop_to_reference(record, reference_seq, min_coverage=0.5):
    """Crop a deposited chain down to the residues a reference sequence also covers.

    The reference is the trajectory's own residue list (`reference_seqs_from_h5`), so this is
    what makes an mdCATH *domain* -- a slice of a chain -- come back as that slice, and what
    trims crystallographic tags and unresolved-loop mismatches for ATLAS. Raises if fewer than
    ``min_coverage`` of the reference's residues could be matched, since a structure that
    disagrees that badly is not the one the trajectory was run on.
    """
    if not reference_seq:
        return record

    ref_idx, rec_idx = align_sequences(reference_seq, record["seq"])
    coverage = len(ref_idx) / len(reference_seq)
    if coverage < min_coverage:
        raise ValueError(
            "{}_{}: deposited chain matches only {:.0%} of the {}-residue reference sequence".format(
                record["pdb_id"], record["chain_id"], coverage, len(reference_seq)))

    cropped = subset_record(record, rec_idx)
    # Remember which reference positions survived, so callers scoring a model whose output is
    # indexed by the *reference* (DynamicMPNN, which still runs on trajectory frames) can line
    # its predictions up with these coordinates without re-aligning.
    cropped["reference_index"] = ref_idx
    cropped["reference_seq"] = reference_seq
    return cropped


class _ResidueSelect(PDB.Select):
    """Write out exactly the residues of a record, first altloc only."""

    def __init__(self, residues):
        self.keep = {id(residue) for residue in residues}

    def accept_residue(self, residue):
        return id(residue) in self.keep

    def accept_atom(self, atom):
        return atom.get_altloc() in (" ", "A")


def write_record_pdb(record, path):
    """Write a record back out as a single-model, single-chain PDB.

    Keeps the deposited side chains (so DSSP's secondary-structure assignment and MapDiff's
    CB-dependent features see a real structure) and the deposited residue numbering (gaps
    included -- `data.generate_graph_cath.get_struc2ndRes` matches Biopython residues to DSSP
    keys by (chain, resseq), and both sides read the same file).
    """
    structure = record["residues"][0].get_parent().get_parent().get_parent()
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(path, select=_ResidueSelect(record["residues"]))
    return path


def reference_seqs_from_h5(h5_path, protein_ids=None):

    wanted = set(protein_ids) if protein_ids is not None else None
    sequences = {}

    def visit(name, obj):
        if not (hasattr(obj, "keys") and "residues" in obj and "coordinates" in obj):
            return
        protein_id = name.split("/")[0]
        if (wanted is not None and protein_id not in wanted) or protein_id in sequences:
            return
        sequences[protein_id] = "".join(
            three_to_one(raw.decode().strip()[:3]) or "X" for raw in obj["residues"][:])

    with h5py.File(h5_path, "r") as h5_file:
        h5_file.visititems(visit)
    return sequences


def load_relaxed_structures(protein_ids, cache_dir="./pdb_cache", reference_seqs=None,
                            min_coverage=0.5, num_workers=8, verbose=True):
    protein_ids = list(dict.fromkeys(protein_ids))
    reference_seqs = reference_seqs or {}

    # One download per *pdb code*, not per protein id: mdCATH routinely has several domains
    # of the same entry, and ATLAS several chains.
    pdb_codes = {}
    for protein_id in protein_ids:
        pdb_id, chain_id = parse_structure_id(protein_id)
        pdb_codes[protein_id] = (pdb_id, chain_id)

    paths, failures = {}, {}
    unique_codes = sorted({pdb_id for pdb_id, _ in pdb_codes.values()})

    def fetch(pdb_id):
        try:
            return pdb_id, fetch_structure_file(pdb_id, cache_dir), None
        except Exception as exc:  # noqa: BLE001 -- any failure here just skips the protein
            return pdb_id, None, str(exc)

    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as pool:
        results = pool.map(fetch, unique_codes)
        if verbose:
            from tqdm import tqdm
            results = tqdm(results, total=len(unique_codes), desc="downloading relaxed PDBs")
        for pdb_id, path, error in results:
            if path is None:
                failures[pdb_id] = error
            else:
                paths[pdb_id] = path

    records = {}
    iterator = protein_ids
    if verbose:
        from tqdm import tqdm
        iterator = tqdm(protein_ids, desc="parsing relaxed PDBs")
    for protein_id in iterator:
        pdb_id, chain_id = pdb_codes[protein_id]
        if pdb_id not in paths:
            failures[protein_id] = failures.get(pdb_id, "download failed")
            continue
        try:
            record = load_chain_backbone(paths[pdb_id], pdb_id, chain_id)
            record = crop_to_reference(record, reference_seqs.get(protein_id), min_coverage=min_coverage)
        except Exception as exc:  # noqa: BLE001
            failures[protein_id] = str(exc)
            continue
        record["protein_id"] = protein_id
        records[protein_id] = record

    if verbose and failures:
        print("relaxed PDB loading: {}/{} protein ids unavailable".format(len(failures), len(protein_ids)))
    return records, failures

    if not coords:
        raise ValueError("bad_seq")

    if any(seq_char not in letter_to_num.keys() for seq_char in seq):
        raise ValueError("bad_seq")
    atom_dict = {
        'name': (pdb_chain_id, -1),
        'seq': seq,
        'coords': np.array(coords)
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

import os

import numpy as np

from relaxed_structures import load_relaxed_structures, reference_seqs_from_h5


def load_raw_split(group_ids, h5_path, cache_dir=os.path.join("..", "cif_cache"),
                    min_coverage=0.5, num_workers=8, verbose=True):
    """Raw backbone records for one split.

    Returns a list of ``{'name': (domain_id, -1), 'seq': seq, 'coords': (R, 4, 3) ndarray}``
    dicts, one per domain id that downloaded, aligned, and matched the H5 reference length.
    """
    group_ids = list(dict.fromkeys(group_ids))

    reference_seqs = reference_seqs_from_h5(h5_path, protein_ids=group_ids)
    missing_from_h5 = [pid for pid in group_ids if pid not in reference_seqs]
    if missing_from_h5 and verbose:
        print("{}/{} domain ids not found in {}; skipping: {}".format(
            len(missing_from_h5), len(group_ids), h5_path, missing_from_h5[:5]))

    wanted = [pid for pid in group_ids if pid in reference_seqs]
    records, failures = load_relaxed_structures(
        wanted,
        cache_dir=cache_dir,
        reference_seqs=reference_seqs,
        min_coverage=min_coverage,
        num_workers=num_workers,
        verbose=verbose,
    )

    raw = []
    for pid in wanted:
        if pid not in records:
            continue  # already in `failures`, logged by load_relaxed_structures
        record = records[pid]
        ref_len = len(reference_seqs[pid])
        if len(record["seq"]) != ref_len:
            if verbose:
                print("Skipping {}: aligned crop has {} residues, H5 reference has {} "
                      "(crystal structure is missing density inside the domain's range)"
                      .format(pid, len(record["seq"]), ref_len))
            continue
        raw.append({
            "name": (pid, -1),
            "seq": record["seq"],
            "coords": np.array(record["coords"]),
        })

    if verbose:
        for pid, reason in failures.items():
            print("Skipping {}: {}".format(pid, reason))
        print("load_raw_split: {}/{} domains usable".format(len(raw), len(group_ids)))

    return raw


def main(index_path, h5_path, use_pdbs=False):

    wandb.login(key=os.getenv("WANDB_KEY"))
    run = wandb.init(
        entity="jenslundsgaard7-uw-madison",
        project="SheafProtein",
        name="gvp",
        config={
            "index_path":index_path
        },
    )
    h5_path = osp.join("..",h5_path)
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



    train_raw = load_raw_split(train_groups, h5_path, cache_dir=cache_dir)
    val_raw   = load_raw_split(val_groups,   h5_path, cache_dir=cache_dir)
    test_raw  = load_raw_split(test_groups,  h5_path, cache_dir=cache_dir)


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


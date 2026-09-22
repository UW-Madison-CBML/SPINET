"""
Load *relaxed* (experimentally deposited) structures from the RCSB PDB, given the protein
ids our datasets are keyed by.

`comparisons/{MapDiff,PiFold}` need the same thing: "for this set of pdb ids, give me the
deposited structure, the sequence it resolves, and its backbone coordinates". Both train and
evaluate on those structures directly. `lib/traj_frames.py` is the alternative source they
can be pointed at instead -- one MD frame per protein, same record format -- which is what
their `--structure-source frame` uses. DynamicMPNN does not use this module at all: it
trains on MD conformer ensembles straight out of the trajectory store, since that ensemble
input *is* the thing being benchmarked.

Two id conventions are in play, both handled by `parse_structure_id`:

* ATLAS  -- ``<pdb>_<chain>``   e.g. ``1a0a_A`` (see scripts/load_dynamics.py's ``__main__``,
  which builds the `pdb` column as ``x[:4] + "_" + x[-1]``).
* mdCATH -- CATH domain ids     e.g. ``1a0aA02``: 4-char pdb code, 1-char chain, 2-digit
  domain number (the `domain` column of mdcath_320_0_topology_split.csv).

Cropping to the simulated region
--------------------------------
A deposited chain is not the same residue set the MD trajectory covers: crystal structures
have unresolved loops and expression tags, and an mdCATH *domain* is only a slice of its
chain. `crop_to_reference` pairwise-aligns the deposited chain's sequence against a reference
sequence (the trajectory's own residue list, via `reference_seqs_from_h5`) and keeps only the
aligned residues. That way every model -- sheaf, MapDiff, PiFold, DynamicMPNN -- sees the
same residues for a given protein, which is the whole point of the comparison. (A trajectory
frame needs none of this: it already spans exactly those residues.)

Network access is required (files.rcsb.org); downloads are cached on disk by pdb code, so a
job only pays for them once.
"""
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from Bio import Align, PDB
from Bio.Data import IUPACData

# Ground-truth order of the atom axis everywhere in this repo (scripts/load_dynamics.py's
# BACKBONE_ATOMS). Duplicated rather than imported because `load_dynamics` pulls in mdtraj,
# which is not installed in every comparison model's image.
BACKBONE_ATOMS = ("CA", "N", "C", "O")

# Force-field/protonation-state and modified-residue variants, mapped onto canonical names.
# Mirrors `lib.residue_classifier_dataset.ResidueClassifierDataset.RESIDUE_ALIASES` plus the
# handful of modified residues that show up in deposited structures.
RESIDUE_ALIASES = {"HSD": "HIS", "HSE": "HIS", "HSP": "HIS", "HID": "HIS",
                   "HIE": "HIS", "HIP": "HIS", "ASH": "ASP", "GLH": "GLU",
                   "LYN": "LYS", "CYM": "CYS", "CYX": "CYS", "MSE": "MET",
                   "SEC": "CYS", "PYL": "LYS"}

THREE_TO_ONE = {name.upper(): letter for name, letter in IUPACData.protein_letters_3to1.items()}

RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.{ext}"

# RCSB asks for a contactable user agent on bulk programmatic access.
_HEADERS = {"User-Agent": "cmikulski@wisc.edu"} # hide!


def three_to_one(resname):
    """Canonical 1-letter code for a 3-letter residue name, or None if not a standard residue."""
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
    """``{protein_id: 1-letter sequence}`` read from a trajectory store's `residues` datasets.

    Works for both shared stores (`atlas_data.h5`, `mdcath_spinet_320_0.h5`): every trajectory
    group holds a `residues` array of 3-letter names, and the top-level group name is the
    protein id the split csvs use.
    """
    import h5py

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
    """Load deposited structures for a set of protein ids.

    This is the entry point every comparison model uses. Returns
    ``(records, failures)`` where ``records`` maps protein id -> the record dict described in
    `load_chain_backbone` (cropped to ``reference_seqs[protein_id]`` when one is supplied) and
    ``failures`` maps protein id -> the reason it could not be loaded.

    Downloads run on a small thread pool (they are entirely network-bound); parsing is serial
    because Biopython's structure objects are not worth shipping between processes.
    """
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

"""
Load a single *MD trajectory frame* per protein as a structure record.

`comparisons/{MapDiff,PiFold}` are static-structure inverse-folding models, so by default
they train and evaluate on each protein's relaxed (deposited) PDB entry
(`lib/relaxed_pdb.py`). This module is the alternative structure source: one frame drawn
from the protein's MD trajectory instead. It exists so the question "how much of the
comparison is driven by the *kind* of structure the model sees?" can be answered by
re-running the same model on the same splits with nothing but the coordinates swapped.

`load_frame_structures` deliberately mirrors `relaxed_pdb.load_relaxed_structures`: same
call shape, same ``(records, failures)`` return, and records carrying the same keys
(including the Biopython ``residues`` objects `relaxed_pdb.write_record_pdb` needs). A
caller therefore swaps one function call and nothing else.

Differences a frame record carries, all of them intrinsic to MD rather than to this code:

* **Backbone only.** The trajectory store keeps `BACKBONE_ATOMS` (CA, N, C, O) and nothing
  else, so there are no side chains and no CB. MapDiff handles this already -- its
  `get_receptor_inference` requires only CA/N/C and masks O/CB, and `place_missing_cb` /
  `place_missing_o` fill the rest -- but DSSP's solvent accessibility is computed on a
  backbone-only chain and so is not comparable, residue for residue, to the SASA of a
  deposited structure. Secondary-structure assignment itself only needs N/C/O and is fine.
* **No cropping.** The trajectory's residue list *is* the reference `relaxed_pdb` crops a
  deposited chain down to, so a frame already spans exactly the simulated residues (for
  mdCATH, exactly the domain). ``reference_index`` is therefore the identity.
* **Units.** `scripts/load_dynamics.py` writes `traj.xyz` straight out of mdtraj, which is
  in NANOMETRES. Deposited structures are in Angstroms, so frames are converted here --
  without it every distance feature in both models would land in the lowest RBF bin (the
  same bug that cost DynamicMPNN ~28 points of recovery; see comparisons/DynamicMPNN/train.py).

Frame choice is random but *deterministic*: the per-protein RNG is seeded from
``(seed, protein_id)``, so a rerun, a resumed job and a second model all draw the same frame
for a given protein and seed, and the draw does not depend on how many proteins are in the
split or what order they are iterated in. Pass ``frame_index`` to pin one frame index for
every protein instead (``0`` = the first frame, i.e. closest to the deposited starting
structure).
"""
import os
import sys
import zlib

import numpy as np
from Bio import PDB

# Flat import, the convention every consumer of lib/ uses (the comparison jobs copy these
# modules in side by side); fall back to this file's own directory for `import lib.traj_frames`.
try:
    from relaxed_pdb import BACKBONE_ATOMS, RESIDUE_ALIASES, three_to_one
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from relaxed_pdb import BACKBONE_ATOMS, RESIDUE_ALIASES, three_to_one

# mdtraj writes nanometres; every model in this repo expects Angstroms. See the module
# docstring, and comparisons/DynamicMPNN/train.py's NM_TO_ANGSTROM for the same conversion.
NM_TO_ANGSTROM = 10.0

# The two datasets every trajectory group must carry to be usable here. Matches
# `lib.residue_classifier_dataset.ResidueClassifierDataset.REQUIRED_DATASETS`, minus the
# features only the sheaf model reads.
REQUIRED_DATASETS = ("coordinates", "residues")

# Conventional backbone atom order for a written-out PDB, as distinct from BACKBONE_ATOMS
# (the order the trajectory store's coordinate axis is in).
PDB_ATOM_ORDER = ("N", "CA", "C", "O")


def index_h5_groups(h5_file, protein_ids=None):
    """``{protein_id: group path}`` for every usable trajectory group in an open store.

    The top-level group name is the protein id the split csvs use, matching
    `relaxed_pdb.reference_seqs_from_h5` and comparisons/DynamicMPNN/train.py.
    """
    wanted = set(protein_ids) if protein_ids is not None else None
    lookup = {}

    def visit(name, obj):
        if not (hasattr(obj, "keys") and all(ds in obj for ds in REQUIRED_DATASETS)):
            return
        protein_id = name.split("/")[0]
        if wanted is not None and protein_id not in wanted:
            return
        lookup.setdefault(protein_id, name)

    h5_file.visititems(visit)
    return lookup


def resolve_group(protein_id, group_lookup):
    """Case-insensitive group lookup -- the stores are not consistent about id casing."""
    for candidate in (protein_id, protein_id.upper(), protein_id.lower()):
        if candidate in group_lookup:
            return group_lookup[candidate]
    return None


def pick_frame(num_frames, protein_id, seed, frame_index=None):
    """Which frame of this protein's trajectory to use.

    Seeded per protein rather than from a single stream, so the frame a protein gets depends
    only on ``(seed, protein_id)`` -- not on split membership or iteration order. That is what
    lets a val run and a test run, or MapDiff and PiFold, agree on the frame per protein.
    """
    if frame_index is not None:
        if not -num_frames <= frame_index < num_frames:
            raise IndexError("frame_index {} out of range for {} ({} frames)".format(
                frame_index, protein_id, num_frames))
        return int(frame_index % num_frames)
    rng = np.random.default_rng([int(seed), zlib.crc32(protein_id.encode())])
    return int(rng.integers(num_frames))


def _build_residues(resnames, coords, protein_id, chain_id="A"):
    """A real Biopython Structure/Model/Chain holding one frame's backbone.

    Built rather than parsed so a frame record can be handed to
    `relaxed_pdb.write_record_pdb` unchanged -- that walks ``residues[0]`` up to its parent
    structure, so the residues have to be attached to one.
    """
    structure = PDB.Structure.Structure(protein_id)
    model = PDB.Model.Model(0)
    chain = PDB.Chain.Chain(chain_id)
    structure.add(model)
    model.add(chain)

    residues = []
    for i, (resname, residue_xyz) in enumerate(zip(resnames, coords)):
        # Number from 1 with no gaps: a trajectory has no unresolved residues, and DSSP
        # matching in MapDiff's `get_struc2ndRes` keys on (chain, resseq) read back out of
        # this same file, so any consistent numbering works.
        residue = PDB.Residue.Residue((" ", i + 1, " "), resname, "")
        # `coords` is in BACKBONE_ATOMS (CA, N, C, O) order, but write them out in the
        # conventional N, CA, C, O order -- mkdssp is happier with a canonical backbone, and
        # MapDiff's `get_receptor_inference` looks atoms up by name either way.
        for atom_name in PDB_ATOM_ORDER:
            atom_xyz = residue_xyz[BACKBONE_ATOMS.index(atom_name)]
            residue.add(PDB.Atom.Atom(
                name=atom_name,
                coord=np.asarray(atom_xyz, dtype=np.float32),
                bfactor=0.0,
                occupancy=1.0,
                altloc=" ",
                fullname=" {:<3}".format(atom_name),
                serial_number=0,
                element=atom_name[0],
            ))
        chain.add(residue)
        residues.append(residue)
    return residues


def load_frame_structures(protein_ids, traj_h5_path, seed=0, frame_index=None, verbose=True):
    """One MD frame per protein, in `relaxed_pdb.load_relaxed_structures`'s record format.

    Returns ``(records, failures)``: ``records`` maps protein id -> a record dict with the
    same keys `relaxed_pdb.load_chain_backbone` produces (``coords`` is ``(R, 4, 3)`` in
    `BACKBONE_ATOMS` order and Angstroms), plus ``frame_index`` / ``num_frames`` for
    provenance; ``failures`` maps protein id -> why it could not be loaded.

    Needs no network access -- everything comes out of the trajectory store.
    """
    import h5py

    protein_ids = list(dict.fromkeys(protein_ids))
    records, failures = {}, {}

    with h5py.File(traj_h5_path, "r") as h5_file:
        group_lookup = index_h5_groups(h5_file)

        iterator = protein_ids
        if verbose:
            from tqdm import tqdm
            iterator = tqdm(protein_ids, desc="reading trajectory frames")

        for protein_id in iterator:
            group_name = resolve_group(protein_id, group_lookup)
            if group_name is None:
                failures[protein_id] = "no trajectory group in {}".format(traj_h5_path)
                continue

            try:
                coords_ds = h5_file[group_name + "/coordinates"]  # R, T, num_atoms, 3 (nm)
                num_frames = coords_ds.shape[1]
                if num_frames == 0:
                    raise ValueError("trajectory has no frames")

                chosen = pick_frame(num_frames, protein_id, seed, frame_index)
                coords = np.asarray(coords_ds[:, chosen], dtype=np.float32) * NM_TO_ANGSTROM

                raw_residues = h5_file[group_name + "/residues"][:]
                if len(raw_residues) != coords.shape[0]:
                    raise ValueError("{} residues but {} coordinate rows".format(
                        len(raw_residues), coords.shape[0]))

                resnames, letters = [], []
                for raw in raw_residues:
                    name = raw.decode().strip()[:3].upper()
                    canonical = RESIDUE_ALIASES.get(name, name)
                    letter = three_to_one(canonical)
                    if letter is None:
                        raise ValueError("non-standard residue {!r}".format(name))
                    resnames.append(canonical)
                    letters.append(letter)

                if not np.isfinite(coords).all():
                    raise ValueError("frame {} has non-finite coordinates".format(chosen))
            except Exception as exc:  # noqa: BLE001 -- any failure here just skips the protein
                failures[protein_id] = str(exc)
                continue

            residues = _build_residues(resnames, coords, protein_id)
            num_residues = len(resnames)
            records[protein_id] = {
                "protein_id": protein_id,
                "pdb_id": protein_id.upper(),
                "chain_id": "A",
                "seq": "".join(letters),
                "resnames": resnames,
                "res_ids": [residue.id[1] for residue in residues],
                "coords": coords,
                # A trajectory frame resolves every residue it simulates, so nothing is
                # masked and the reference correspondence is the identity -- unlike a
                # deposited chain, which `relaxed_pdb.crop_to_reference` has to align.
                "mask": np.ones(num_residues, dtype=bool),
                "residues": residues,
                "path": str(traj_h5_path),
                "reference_index": np.arange(num_residues),
                "reference_seq": "".join(letters),
                "frame_index": chosen,
                "num_frames": int(num_frames),
            }

    if verbose and failures:
        print("trajectory frame loading: {}/{} protein ids unavailable".format(
            len(failures), len(protein_ids)))
    return records, failures

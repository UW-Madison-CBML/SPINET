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
* **Units.** The stores do not agree: ATLAS (`scripts/load_dynamics.py` writes `traj.xyz`
  straight out of mdtraj) is in NANOMETRES, while the mdCATH store is already in Angstroms.
  Deposited structures are in Angstroms, so each frame's units are read off its own CA-CA
  spacing (`detect_coord_scale`) and converted when needed. Getting it wrong either way is
  silent: nm pushes every distance feature into the lowest RBF bin (the bug that cost
  DynamicMPNN ~28 points of recovery), and scaling an Angstrom store by 10 pushes them all
  into the top bin (see comparisons/DynamicMPNN/train.py, which does the same detection).

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
    import dataset_splits
    from relaxed_pdb import BACKBONE_ATOMS, RESIDUE_ALIASES, three_to_one
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import dataset_splits
    from relaxed_pdb import BACKBONE_ATOMS, RESIDUE_ALIASES, three_to_one

# ATLAS is in nanometres (mdtraj), mdCATH in Angstroms; every model in this repo expects
# Angstroms. See the module docstring, and comparisons/DynamicMPNN/train.py's
# detect_coord_scale, which this mirrors.
NM_TO_ANGSTROM = 10.0

# A backbone CA-CA virtual bond is ~3.80 A / 0.380 nm in every folded protein, so measuring it
# identifies a frame's units.
CA_CA_ANGSTROM = 3.80
CA_CA_TOLERANCE = 0.30

# The two datasets every trajectory group must carry to be usable here. Matches
# `lib.residue_classifier_dataset.ResidueClassifierDataset.REQUIRED_DATASETS`, minus the
# features only the sheaf model reads.
REQUIRED_DATASETS = ("coordinates", "residues")

# Conventional backbone atom order for a written-out PDB, as distinct from BACKBONE_ATOMS
# (the order the trajectory store's coordinate axis is in).
PDB_ATOM_ORDER = ("N", "CA", "C", "O")


def index_h5_groups(h5_file, protein_ids=None, required=REQUIRED_DATASETS, all_groups=False):
    """``{protein_id: group path}`` for every usable trajectory group in an open store.

    The top-level group name is the protein id the split csvs use, matching
    `relaxed_pdb.reference_seqs_from_h5` and comparisons/DynamicMPNN/train.py. A group is usable
    if it holds every dataset in ``required``. With ``all_groups`` the values are lists of every
    usable group under the protein (in visit order) instead of the first one.
    """
    wanted = set(protein_ids) if protein_ids is not None else None
    lookup = {}

    def visit(name, obj):
        if not (hasattr(obj, "keys") and all(ds in obj for ds in required)):
            return
        protein_id = name.split("/")[0]
        if wanted is not None and protein_id not in wanted:
            return
        if all_groups:
            lookup.setdefault(protein_id, []).append(name)
        else:
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


def detect_coord_scale(frame_coords):
    """Factor converting one frame's ``(R, num_atoms, 3)`` coordinates to Angstroms.

    Returns 1.0 for Angstroms and NM_TO_ANGSTROM for nanometres, and raises if the median
    consecutive CA-CA distance matches neither -- both wrong guesses corrupt features silently.
    """
    ca = np.asarray(frame_coords[:, BACKBONE_ATOMS.index("CA"), :], dtype=np.float64)
    if ca.shape[0] < 2:
        raise ValueError("cannot detect coordinate units from a single-residue chain")
    spacing = float(np.median(np.linalg.norm(np.diff(ca, axis=0), axis=1)))
    for scale in (1.0, NM_TO_ANGSTROM):
        if abs(spacing * scale - CA_CA_ANGSTROM) <= CA_CA_TOLERANCE:
            return scale
    raise ValueError(
        "median consecutive CA-CA distance is {:.4f}, neither {} A nor {} nm; refusing to "
        "guess the coordinate units".format(spacing, CA_CA_ANGSTROM, CA_CA_ANGSTROM / NM_TO_ANGSTROM))


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


def parse_residues(raw_residues):
    """``(resnames, letters)`` from a trajectory group's ``residues`` dataset.

    Raises on any residue with no one-letter code -- the caller decides whether that skips the
    protein.
    """
    resnames, letters = [], []
    for raw in raw_residues:
        name = (raw.decode() if isinstance(raw, bytes) else str(raw)).strip()[:3].upper()
        canonical = RESIDUE_ALIASES.get(name, name)
        letter = three_to_one(canonical)
        if letter is None:
            raise ValueError("non-standard residue {!r}".format(name))
        resnames.append(canonical)
        letters.append(letter)
    return resnames, letters


def frame_record(protein_id, resnames, letters, coords, frame_index, num_frames, path):
    """One frame as a `relaxed_pdb.load_chain_backbone`-shaped record (``coords`` in Angstroms,
    `BACKBONE_ATOMS` order), so `relaxed_pdb.write_record_pdb` can write it out unchanged."""
    residues = _build_residues(resnames, coords, protein_id)
    num_residues = len(resnames)
    seq = "".join(letters)
    return {
        "protein_id": protein_id,
        "pdb_id": protein_id.upper(),
        "chain_id": "A",
        "seq": seq,
        "resnames": resnames,
        "res_ids": [residue.id[1] for residue in residues],
        "coords": coords,
        # A trajectory frame resolves every residue it simulates, so nothing is masked and
        # the reference correspondence is the identity -- unlike a deposited chain, which
        # `relaxed_pdb.crop_to_reference` has to align.
        "mask": np.ones(num_residues, dtype=bool),
        "residues": residues,
        "path": str(path),
        "reference_index": np.arange(num_residues),
        "reference_seq": seq,
        "frame_index": int(frame_index),
        "num_frames": int(num_frames),
    }


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
                coords_ds = h5_file[group_name + "/coordinates"]  # R, T, num_atoms, 3 (nm or A)
                num_frames = coords_ds.shape[1]
                if num_frames == 0:
                    raise ValueError("trajectory has no frames")

                chosen = pick_frame(num_frames, protein_id, seed, frame_index)
                coords = np.asarray(coords_ds[:, chosen], dtype=np.float32)
                coords = coords * np.float32(detect_coord_scale(coords))

                resnames, letters = parse_residues(h5_file[group_name + "/residues"][:])
                if len(resnames) != coords.shape[0]:
                    raise ValueError("{} residues but {} coordinate rows".format(
                        len(resnames), coords.shape[0]))

                if not np.isfinite(coords).all():
                    raise ValueError("frame {} has non-finite coordinates".format(chosen))
            except Exception as exc:  # noqa: BLE001 -- any failure here just skips the protein
                failures[protein_id] = str(exc)
                continue

            records[protein_id] = frame_record(
                protein_id, resnames, letters, coords, chosen, num_frames, traj_h5_path)

    if verbose and failures:
        print("trajectory frame loading: {}/{} protein ids unavailable".format(
            len(failures), len(protein_ids)))
    return records, failures


# ---------------------------------------------------------------------------------------------
# Every frame of every trajectory
#
# The single-frame loader above answers "relaxed vs. one MD frame". What follows serves the
# *all-frames* setup instead: a static model is trained on every (protein, frame) pair -- each
# frame is just another structure with the same sequence -- and evaluated per protein by running
# it on every frame and averaging the logits across frames. A protein's frames are pooled over
# every trajectory (replica) the sheaf model uses for it. Shared by the comparisons'
# dataloaders (comparisons/MapDiff/dataloader/traj_dataset.py,
# comparisons/PiFold/API/traj_dataset.py) so they all see the same frames.
# ---------------------------------------------------------------------------------------------

def sheaf_frames(num_frames, ds_name):
    """The frame indices the sheaf model trains on for a ``num_frames``-long trajectory.

    Mirrors `ResidueClassifierDataset.build_dynamic_index` as scripts/train_residue_classifier.py
    configures it (lib/dataset_splits.WINDOW_LEN / WINDOW_STEP / TRAJ_LEN): windows of WINDOW_LEN
    frames starting at 0, WINDOW_STEP, ... within the first TRAJ_LEN frames, so the frames covered
    are ``range(last_start + WINDOW_LEN)``. Empty when the trajectory is shorter than one window
    -- the sheaf model skips that trajectory.
    """
    cap = dataset_splits.TRAJ_LEN[ds_name.lower()]
    length = min(num_frames, cap) if cap is not None else num_frames
    if length < dataset_splits.WINDOW_LEN:
        return []
    last_start = ((length - dataset_splits.WINDOW_LEN) // dataset_splits.WINDOW_STEP) * dataset_splits.WINDOW_STEP
    return list(range(last_start + dataset_splits.WINDOW_LEN))


def select_frames(num_frames, stride=1, max_frames=None):
    """Which frame indices of a ``num_frames``-long trajectory to use.

    Every ``stride``-th frame, then -- if that is still more than ``max_frames`` -- thinned to
    ``max_frames`` evenly spaced ones. Deterministic, so featurization, training and evaluation
    agree on the frame set without passing it around.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1, got {}".format(stride))
    # Returns positions into whatever frame list the caller thins (TrajectoryStore.index applies
    # it to `sheaf_frames`), so a stride of 1 and no max_frames keeps that list whole.
    indices = np.arange(0, num_frames, stride)
    if max_frames is not None and 0 < max_frames < len(indices):
        indices = indices[np.linspace(0, len(indices) - 1, max_frames).round().astype(int)]
    return [int(i) for i in indices]


# A frame key ``(group path, frame index)`` as a file name, for featurizers that write one file per
# frame (MapDiff's data/graphs_from_traj.py): ``<group below the protein id, '/' -> '+'>__<frame>.pt``,
# e.g. ``1a0a_A/temp_1/1a0a_A_R1`` frame 12 -> ``temp_1+1a0a_A_R1__00012.pt``. A group that *is*
# the protein's top-level group is written as ``root``.
_ROOT_TOKEN = "root"


def frame_key_filename(protein_id, frame_key, suffix=".pt"):
    group_name, frame_index = frame_key
    relative = group_name[len(protein_id):].lstrip("/") if group_name.startswith(protein_id) else group_name
    if "+" in relative or "__" in relative:
        raise ValueError("cannot encode trajectory group {!r} in a file name".format(group_name))
    token = relative.replace("/", "+") or _ROOT_TOKEN
    return "{}__{:05d}{}".format(token, int(frame_index), suffix)


def parse_frame_filename(protein_id, filename, suffix=".pt"):
    """Inverse of `frame_key_filename`: ``(group path, frame index)``."""
    stem = filename[:-len(suffix)] if suffix and filename.endswith(suffix) else filename
    token, frame = stem.rsplit("__", 1)
    group_name = protein_id if token == _ROOT_TOKEN else "{}/{}".format(protein_id, token.replace("+", "/"))
    return group_name, int(frame)


class TrajectoryStore:
    """Lazy reader over a trajectory store, safe to share with DataLoader workers.

    `index` reads only metadata (residue names, frame counts, and frame 0 for the unit check);
    coordinates are read per request with `read_frames`. The h5 handle is (re)opened per
    process, because an h5py handle inherited across ``fork`` is not safe to read from.
    """

    def __init__(self, h5_path):
        self.h5_path = str(h5_path)
        self._handle = None
        self._pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        state["_pid"] = None
        return state

    @property
    def h5(self):
        if self._handle is None or self._pid != os.getpid():
            import h5py
            self._handle = h5py.File(self.h5_path, "r")
            self._pid = os.getpid()
        return self._handle

    def close(self):
        if self._handle is not None and self._pid == os.getpid():
            self._handle.close()
        self._handle = None
        self._pid = None

    def index(self, protein_ids, ds_name, stride=1, max_frames=None, max_length=None, verbose=True):
        """``(entries, failures)`` for ``protein_ids``, in input order.

        Uses the trajectories and frames the sheaf model trains on (scripts/train_residue_classifier.py
        -> `ResidueClassifierDataset`): *every* group under a protein holding all of
        `dataset_splits.SHEAF_REQUIRED_DATASETS` is its own trajectory (e.g. several replicas),
        each contributing the frames its own windows cover (`sheaf_frames`, which needs
        ``ds_name``). A group shorter than one window is skipped, as the sheaf model skips it; a
        protein with no usable group fails. ``stride`` thins each group's frames and
        ``max_frames`` then caps the protein's pooled frames (evenly spaced, `select_frames`);
        leave both at their defaults to use exactly the sheaf frames.

        ``entries`` maps protein id -> ``{'seq', 'resnames', 'groups', 'frames'}``:

        * ``groups`` -- ``{group path: {'num_frames', 'scale'}}`` for each usable group, ``scale``
          being the nm -> A factor (detected once per group on frame 0);
        * ``frames`` -- frame *keys* ``(group path, frame index)``, group by group in store
          order, which is what `read_frames` / `frame_records` take.

        ``failures`` maps protein id -> why it was dropped; ``max_length`` drops proteins longer
        than that many residues. All groups of a protein must agree on its residues.
        """
        protein_ids = list(dict.fromkeys(protein_ids))
        group_lookup = index_h5_groups(self.h5, required=dataset_splits.SHEAF_REQUIRED_DATASETS,
                                       all_groups=True)
        entries, failures = {}, {}

        iterator = protein_ids
        if verbose:
            from tqdm import tqdm
            iterator = tqdm(protein_ids, desc="indexing trajectories")

        for protein_id in iterator:
            # Exact match, not `resolve_group`'s case-insensitive one: ResidueClassifierDataset
            # keeps a group only if `name.split("/")[0] in split`, so a casing mismatch drops the
            # protein there and must drop it here too.
            group_names = group_lookup.get(protein_id)
            if not group_names:
                failures[protein_id] = "no trajectory group with all of {} in {}".format(
                    dataset_splits.SHEAF_REQUIRED_DATASETS, self.h5_path)
                continue
            try:
                seq, resnames, groups, frames, skipped = None, None, {}, [], {}
                for group_name in group_names:
                    coords_ds = self.h5[group_name + "/coordinates"]  # R, T, num_atoms, 3
                    num_residues, num_frames = coords_ds.shape[:2]
                    group_resnames, letters = parse_residues(self.h5[group_name + "/residues"][:])
                    if len(group_resnames) != num_residues:
                        raise ValueError("{}: {} residues but {} coordinate rows".format(
                            group_name, len(group_resnames), num_residues))
                    if seq is None:
                        seq, resnames = "".join(letters), group_resnames
                    elif "".join(letters) != seq:
                        raise ValueError("trajectory groups disagree on the residue sequence "
                                         "({} vs {})".format(group_names[0], group_name))
                    if max_length is not None and num_residues > max_length:
                        raise ValueError("too long ({} > {} residues)".format(num_residues, max_length))
                    covered = sheaf_frames(num_frames, ds_name)
                    if not covered:
                        skipped[group_name] = "{} frames, shorter than one {}-frame window".format(
                            num_frames, dataset_splits.WINDOW_LEN)
                        continue
                    groups[group_name] = {
                        "num_frames": int(num_frames),
                        "scale": float(detect_coord_scale(np.asarray(coords_ds[:, 0], dtype=np.float32))),
                    }
                    frames.extend((group_name, covered[i]) for i in select_frames(len(covered), stride))
                if not groups:
                    raise ValueError("no trajectory long enough for the sheaf model ({})".format(skipped))
            except Exception as exc:  # noqa: BLE001 -- any failure here just skips the protein
                failures[protein_id] = str(exc)
                continue

            entries[protein_id] = {
                "seq": seq,
                "resnames": resnames,
                "groups": groups,
                "frames": [frames[i] for i in select_frames(len(frames), max_frames=max_frames)],
            }

        if verbose and failures:
            print("trajectory indexing: {}/{} protein ids unavailable".format(
                len(failures), len(protein_ids)))
        return entries, failures

    def read_frames(self, entry, frame_keys):
        """``(F, R, 4, 3)`` float32 Angstrom coordinates, `BACKBONE_ATOMS` order, one per
        requested ``(group path, frame index)`` key (in the order given)."""
        frame_keys = [(str(g), int(f)) for g, f in frame_keys]
        out = [None] * len(frame_keys)
        by_group = {}
        for position, (group_name, frame_index) in enumerate(frame_keys):
            by_group.setdefault(group_name, []).append((position, frame_index))
        for group_name, wanted in by_group.items():
            coords_ds = self.h5[group_name + "/coordinates"]
            # h5py fancy indexing wants strictly increasing indices: read sorted+unique, then
            # scatter back into the requested order.
            unique, inverse = np.unique([f for _, f in wanted], return_inverse=True)
            block = np.asarray(coords_ds[:, unique.tolist()], dtype=np.float32)  # R, U, A, 3
            block = block.transpose(1, 0, 2, 3) * np.float32(entry["groups"][group_name]["scale"])
            for (position, _), j in zip(wanted, inverse):
                out[position] = block[j]
        coords = np.ascontiguousarray(np.stack(out))
        if not np.isfinite(coords).all():
            raise ValueError("non-finite coordinates in frames {}".format(frame_keys))
        return coords

    def frame_records(self, protein_id, entry, frame_keys=None):
        """Yield one `frame_record` per frame key (default: ``entry['frames']``) -- what a
        featurizer that writes PDBs (MapDiff's DSSP step) consumes. Each record carries its
        ``group`` alongside ``frame_index``."""
        frame_keys = entry["frames"] if frame_keys is None else frame_keys
        letters = list(entry["seq"])
        for (group_name, frame_index), coords in zip(frame_keys, self.read_frames(entry, frame_keys)):
            record = frame_record(protein_id, entry["resnames"], letters, coords, frame_index,
                                  entry["groups"][group_name]["num_frames"], self.h5_path)
            record["group"] = group_name
            yield record


class FramesPerProteinSampler:
    """Sampler over a flat ``(protein, frame)`` dataset that draws ``frames_per_protein`` frames
    of every protein per epoch.

    ``frame_counts[i]`` is how many frames protein ``i`` contributes, and the flat dataset is
    laid out protein by protein (protein 0's frames first) -- the layout both comparisons'
    per-frame datasets use. With ``frames_per_protein=None`` every frame is used every epoch,
    which makes an epoch ``sum(frame_counts)`` items long; a small value keeps an epoch about as
    long as a single-structure epoch while still cycling through the whole trajectory across
    epochs. Proteins with fewer frames contribute all of them.

    Reseeded per epoch from ``(seed, epoch)``, so a run is reproducible. Pass it as a
    DataLoader's ``sampler=`` (and drop ``shuffle=True``).
    """

    def __init__(self, frame_counts, frames_per_protein=None, shuffle=True, seed=0):
        self.frame_counts = [int(n) for n in frame_counts]
        self.offsets = np.concatenate([[0], np.cumsum(self.frame_counts)[:-1]]).astype(np.int64)
        self.frames_per_protein = frames_per_protein
        self.shuffle = shuffle
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        k = self.frames_per_protein
        return int(sum(n if k is None else min(k, n) for n in self.frame_counts))

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch])
        self.epoch += 1
        k = self.frames_per_protein
        picks = []
        for offset, n in zip(self.offsets, self.frame_counts):
            if k is None or k >= n:
                picks.append(offset + np.arange(n))
            else:
                picks.append(offset + rng.choice(n, size=k, replace=False))
        order = np.concatenate(picks) if picks else np.zeros(0, dtype=np.int64)
        if self.shuffle:
            rng.shuffle(order)
        return iter(order.tolist())


def average_frame_logits(logits, num_frames):
    """Mean over frames of per-residue logits from one protein's frame ensemble.

    ``logits`` is ``(num_frames * R, C)`` with frames stacked frame-major -- what a batch of one
    protein's frames gives back from MapDiff (`Batch.from_data_list`) or PiFold
    (`featurize_GTrans`, no missing atoms in a trajectory frame) -- or already ``(num_frames,
    R, C)``. Returns ``(R, C)``. Works on torch tensors and numpy arrays alike.
    """
    if logits.ndim == 2:
        if logits.shape[0] % num_frames:
            raise ValueError("{} logit rows do not split into {} equal frames".format(
                logits.shape[0], num_frames))
        logits = logits.reshape(num_frames, logits.shape[0] // num_frames, logits.shape[1])
    return logits.mean(0)

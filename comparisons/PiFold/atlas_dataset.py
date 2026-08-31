import os
import gzip
import numpy as np
from tqdm import tqdm

import torch.utils.data as data
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1

from .utils import cached_property

ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
BACKBONE_ATOMS = ('N', 'CA', 'C', 'O')

# protein_letters_3to1 only recognizes the 20 canonical residue names; map a
# few common variants seen in crystal/MD structures onto their canonical letter.
_THREE_TO_ONE_EXTRA = {'MSE': 'M', 'SEC': 'C', 'PYL': 'K', 'HSD': 'H', 'HSE': 'H', 'HSP': 'H'}


def _res_to_one(resname):
    resname = resname.upper()
    if resname in _THREE_TO_ONE_EXTRA:
        return _THREE_TO_ONE_EXTRA[resname]
    return protein_letters_3to1.get(resname)


def parse_pdb_backbone(filepath):
    """Parse a (optionally gzipped) single-chain PDB file.

    Returns a dict with keys 'seq', 'N', 'CA', 'C', 'O' (coords as [L, 3]
    float32 arrays), using the first model and first chain in the file.
    Residues missing any backbone atom, or that aren't standard amino acids
    (waters, ligands, HETATMs), are dropped. Returns None if no usable
    residues are found.
    """
    parser = PDBParser(QUIET=True)
    opener = gzip.open if filepath.endswith('.gz') else open
    with opener(filepath, 'rt') as handle:
        structure = parser.get_structure(os.path.basename(filepath), handle)

    model = next(iter(structure), None)
    if model is None:
        return None

    chain = next(iter(model), None)
    if chain is None:
        return None

    seq = []
    coords = {atom: [] for atom in BACKBONE_ATOMS}
    for residue in chain:
        hetflag, _, _ = residue.id
        if hetflag.strip() != '':
            continue  # skip waters/ligands/other HETATM records

        aa = _res_to_one(residue.get_resname())
        if aa is None:
            continue
        if not all(atom in residue for atom in BACKBONE_ATOMS):
            continue  # incomplete backbone, skip this residue

        seq.append(aa)
        for atom in BACKBONE_ATOMS:
            coords[atom].append(residue[atom].get_coord())

    if len(seq) == 0:
        return None

    return {
        'seq': ''.join(seq),
        **{atom: np.asarray(coords[atom], dtype=np.float32) for atom in BACKBONE_ATOMS},
    }


class ATLAS(data.Dataset):
    """PiFold-compatible dataset built from ATLAS-derived PDB files that have
    already been split into train/valid/test folders on disk (one PDB file
    per MD simulation, e.g. its first frame).

    Expected layout under `path`:
        {path}/train/*.pdb[.gz]
        {path}/{valid,validation,val}/*.pdb[.gz]
        {path}/test/*.pdb[.gz]

    Produces items shaped like API.cath_dataset.CATH: dicts with
    'title', 'seq', 'N', 'CA', 'C', 'O' (and 'category'/'score' for test),
    so it plugs directly into API.featurizer.featurize_GTrans.
    """

    SPLIT_DIRS = {
        'train': ['train'],
        'valid': ['valid', 'validation', 'val'],
        'test': ['test'],
    }

    def __init__(self, path='./', mode='train', max_length=500, data=None):
        self.path = path
        self.mode = mode
        self.max_length = max_length
        if data is None:
            self.data = self.cache_data[mode]
        else:
            self.data = data

    def _resolve_split_dir(self, mode):
        for cand in self.SPLIT_DIRS[mode]:
            cand_path = os.path.join(self.path, cand)
            if os.path.isdir(cand_path):
                return cand_path
        raise FileNotFoundError(
            "Could not find a '{}' split folder (tried {}) under {}".format(
                mode, self.SPLIT_DIRS[mode], self.path)
        )

    @cached_property
    def cache_data(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError("no such directory: {} !!!".format(self.path))

        alphabet_set = set(ALPHABET)
        data_dict = {'train': [], 'valid': [], 'test': []}

        for mode in data_dict:
            split_dir = self._resolve_split_dir(mode)
            files = sorted(
                f for f in os.listdir(split_dir)
                if f.endswith('.pdb') or f.endswith('.pdb.gz')
            )
            for fname in tqdm(files, desc='loading ATLAS/{}'.format(mode)):
                fpath = os.path.join(split_dir, fname)
                parsed = parse_pdb_backbone(fpath)
                if parsed is None:
                    continue

                seq = parsed['seq']
                bad_chars = set(seq).difference(alphabet_set)
                if len(bad_chars) > 0:
                    continue
                if len(seq) > self.max_length:
                    continue

                title = fname
                for suffix in ('.pdb.gz', '.pdb'):
                    if title.endswith(suffix):
                        title = title[:-len(suffix)]
                        break

                entry = {
                    'title': title,
                    'seq': seq,
                    'CA': parsed['CA'],
                    'C': parsed['C'],
                    'O': parsed['O'],
                    'N': parsed['N'],
                    'category': 'ATLAS',
                }
                if mode == 'test':
                    entry['score'] = 100.0

                data_dict[mode].append(entry)

        return data_dict

    def change_mode(self, mode):
        self.mode = mode
        self.data = self.cache_data[mode]

    def __len__(self):
        return len(self.data)

    def get_item(self, index):
        return self.data[index]

    def __getitem__(self, index):
        return self.data[index]

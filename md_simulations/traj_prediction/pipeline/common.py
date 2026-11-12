"""Shared helpers for the refold-and-resimulate pipeline.

Every trajectory that reaches the analysis is reduced to the same backbone
topology (N, CA, C, O per residue, one chain), so that the mdCATH reference
window and the new simulations can be compared atom for atom even though their
sequences differ.
"""
import numpy as np

# Atom order used for every backbone trajectory in this pipeline.
BACKBONE = ("N", "CA", "C", "O")

# Column positions of each backbone atom in the 18-column shard feature array
# (same layout as scripts/mdcath_export_for_pymol.py).
ATOM_COLS = {"N": slice(15, 18), "CA": slice(9, 12), "C": slice(6, 9), "O": slice(12, 15)}

# Protonation-state and force-field residue names mapped to canonical names
# (mirrors RESIDUE_ALIASES in lib/relaxed_pdb.py, without its Biopython import).
RESIDUE_ALIASES = {"HSD": "HIS", "HSE": "HIS", "HSP": "HIS", "HID": "HIS", "HIE": "HIS",
                   "HIP": "HIS", "ASH": "ASP", "GLH": "GLU", "LYN": "LYS", "CYM": "CYS",
                   "CYX": "CYS", "MSE": "MET", "SEC": "CYS", "PYL": "LYS"}
THREE_TO_ONE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
                "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
                "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
                "TYR": "Y", "VAL": "V"}
ONE_TO_THREE = {v: k for k, v in THREE_TO_ONE.items()}


def canonical(resname):
    """Canonical 3-letter name of a residue, e.g. HSD -> HIS."""
    name = resname.strip().upper()
    return RESIDUE_ALIASES.get(name, name)


def three_to_one(resname):
    """1-letter code of a residue, or None when it is not a standard amino acid."""
    return THREE_TO_ONE.get(canonical(resname))


def parse_traj_id(traj_id):
    """Split '<domain>_<temperature>_<replica>' into (domain, temperature K, replica)."""
    domain, temp, rep = traj_id.rsplit("_", 2)
    return domain, int(temp), int(rep)


def backbone_topology(resnames, resids=None):
    """Return an mdtraj Topology with atoms N, CA, C, O for each residue, in one chain."""
    import mdtraj as md
    from mdtraj.core.element import carbon, nitrogen, oxygen
    elements = {"N": nitrogen, "CA": carbon, "C": carbon, "O": oxygen}
    top = md.Topology()
    chain = top.add_chain()
    prev_c = None
    for k, name in enumerate(resnames):
        seq = int(resids[k]) if resids is not None else k + 1
        res = top.add_residue(canonical(name), chain, resSeq=seq)
        atoms = {a: top.add_atom(a, elements[a], res) for a in BACKBONE}
        top.add_bond(atoms["N"], atoms["CA"])
        top.add_bond(atoms["CA"], atoms["C"])
        top.add_bond(atoms["C"], atoms["O"])
        if prev_c is not None:
            top.add_bond(prev_c, atoms["N"])
        prev_c = atoms["C"]
    return top


def to_backbone(traj):
    """Reduce an mdtraj Trajectory to the canonical backbone topology.

    Keeps residues that have all four atoms N, CA, C, O. Water, ions and the
    CHARMM terminal caps (their atoms are named CAY, CY, OY, NT, CAT) drop out.
    """
    import mdtraj as md
    index, resnames, resids = [], [], []
    for res in traj.topology.residues:
        atoms = {a.name: a.index for a in res.atoms}
        if all(name in atoms for name in BACKBONE) and three_to_one(res.name):
            index.extend(atoms[name] for name in BACKBONE)
            resnames.append(res.name)
            resids.append(res.resSeq)
    if not index:
        raise ValueError("no protein backbone atoms found")
    top = backbone_topology(resnames, resids)
    return md.Trajectory(traj.xyz[:, index], top, time=traj.time,
                         unitcell_lengths=traj.unitcell_lengths,
                         unitcell_angles=traj.unitcell_angles)


def sequence_of(top):
    """1-letter sequence of an mdtraj Topology."""
    return "".join(three_to_one(r.name) or "X" for r in top.residues)


def chain_breaks(traj, max_cn_nm=0.16):
    """Indices k where residue k and k+1 are not bonded in frame 0 (C-N distance too long)."""
    top = traj.topology
    c = [a.index for a in top.atoms if a.name == "C"]
    n = [a.index for a in top.atoms if a.name == "N"]
    d = np.linalg.norm(traj.xyz[0, c[:-1]] - traj.xyz[0, n[1:]], axis=1)
    return [int(k) for k in np.where(d > max_cn_nm)[0]]

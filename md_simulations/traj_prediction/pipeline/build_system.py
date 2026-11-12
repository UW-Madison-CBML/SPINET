#!/usr/bin/env python3
"""Step 3a: build a solvated CHARMM22* system from a folded model, as mdCATH did.

mdCATH (Mirarchi et al., Sci. Data 2024) built its systems with HTMD:
  * protonation at pH 7 with hydrogen-bond network optimisation;
  * acetylated N-terminus and N-methylated C-terminus (CHARMM patches ACE, CT3);
  * CHARMM22* force field, TIP3P water;
  * cubic box with at least 9 A of water on each side of the protein;
  * neutralised, then 0.150 M NaCl.

Output (in --out-dir)
---------------------
structure.psf, structure.pdb   The built system.
ff/                            The CHARMM topology and parameter files for OpenMM.
system.json                    Box side (A), atom count, residue count.
"""
import argparse
import inspect
import json
import shutil
from pathlib import Path

import numpy as np

TOPOLOGIES = ["top/top_all22star_prot.rtf", "top/top_water_ions.rtf"]
PARAMETERS = ["par/par_all22star_prot.prm", "par/par_water_ions.prm"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pdb", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--padding", type=float, default=9.0, help="water padding in Angstrom")
    p.add_argument("--salt", type=float, default=0.150, help="NaCl concentration in mol/L")
    p.add_argument("--ph", type=float, default=7.0)
    args = p.parse_args()

    from htmd.builder import charmm
    from htmd.builder.solvate import solvate
    from moleculekit.molecule import Molecule
    from moleculekit.tools.preparation import systemPrepare

    mol = Molecule(str(args.pdb))
    mol.filter("protein")
    mol.remove("name OXT")              # the CT3 cap replaces the terminal carboxylate
    mol = systemPrepare(mol, pH=args.ph)
    mol.set("segid", "P0")
    mol.set("chain", "A")

    # Cubic box: the longest protein dimension plus the padding on both sides.
    xyz = mol.coords[:, :, 0]
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    mol.moveBy(-(lo + hi) / 2.0)
    half = float((hi - lo).max()) / 2.0 + args.padding
    solvated = solvate(mol, minmax=[[-half] * 3, [half] * 3])

    out = args.out_dir
    kwargs = dict(topo=TOPOLOGIES, outdir=str(out), saltconc=args.salt, ionize=True)
    if "param" in inspect.signature(charmm.build).parameters:   # HTMD before 2.x
        kwargs["param"] = PARAMETERS
    built = charmm.build(solvated, **kwargs)

    ff = out / "ff"
    ff.mkdir(exist_ok=True)
    home = Path(charmm.htmdCharmmHome())
    for name in TOPOLOGIES + PARAMETERS:
        shutil.copy(home / name, ff / Path(name).name)

    info = dict(source_pdb=args.pdb.name, box_A=2.0 * half, n_atoms=int(built.numAtoms),
                n_protein_residues=int(len(np.unique(built.resid[built.segid == "P0"]))),
                n_sodium=int((built.resname == "SOD").sum()),
                n_chloride=int((built.resname == "CLA").sum()),
                padding_A=args.padding, salt_M=args.salt, ph=args.ph,
                force_field="CHARMM22*", water="TIP3P", caps="ACE / CT3")
    (out / "system.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()

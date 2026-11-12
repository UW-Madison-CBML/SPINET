#!/usr/bin/env python3
"""Step 3b/3c: equilibrate and run production MD with the mdCATH settings in OpenMM.

mdCATH settings (Mirarchi et al., Sci. Data 2024), reproduced here:
  * 4 fs time step, hydrogen mass 4 amu, bonds to hydrogen constrained (tol. 1e-5);
  * PME electrostatics, 9 A cutoff, van der Waals switched from 7.5 A to 9 A;
  * Langevin thermostat, friction 0.1 /ps;
  * equilibration: NPT (Monte Carlo barostat), 1 atm, 300 K, 20 ns. Harmonic
    restraints on CA atoms (1.0 kcal/mol/A^2) and other protein heavy atoms
    (0.1 kcal/mol/A^2) for the first 10 ns, none for the second 10 ns;
  * production: NVT at the trajectory temperature, one frame every 1 ns. All
    replicas start from the same equilibrated system with new velocities.

Modes
-----
equil  Reads the system from build_system.py, minimises, equilibrates, and
       writes equil_state.xml and equil_summary.json into the build folder.
prod   Reads the build folder (with equil_state.xml) and writes into --out-dir:
         traj.xtc         Protein atoms only. Frame 0 is the start state, then
                          one frame per interval. Molecules are not wrapped.
         top.pdb          Protein topology for traj.xtc.
         log.csv          Temperature and energies every 100 ps.
         md_summary.json  Frames written, ns/day, whether the run was cut short.

Time budget
-----------
--budget-hours is the wall-clock limit for equilibration plus production of one
simulation. Production stops early when the next frame would pass the limit.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import openmm as mm
from openmm import app, unit

KCAL_A2 = 418.4          # 1 kcal/mol/A^2 in kJ/mol/nm^2
SOLVENT = {"TIP3", "HOH", "WAT", "SOD", "CLA", "NA", "CL", "POT"}


def load_build(build_dir):
    """Return (topology, positions, system factory) for a build_system.py folder."""
    psf = app.CharmmPsfFile(str(build_dir / "structure.psf"))
    pdb = app.PDBFile(str(build_dir / "structure.pdb"))
    box = json.loads((build_dir / "system.json").read_text())["box_A"] * unit.angstrom
    psf.setBox(box, box, box)
    ff = sorted(str(f) for f in (build_dir / "ff").iterdir())
    params = app.CharmmParameterSet(*ff)

    def make_system():
        return psf.createSystem(params, nonbondedMethod=app.PME,
                                nonbondedCutoff=0.9 * unit.nanometer,
                                switchDistance=0.75 * unit.nanometer,
                                constraints=app.HBonds, rigidWater=True,
                                hydrogenMass=4.0 * unit.amu)
    return psf.topology, pdb.positions, make_system


def protein_atoms(topology):
    return [a.index for a in topology.atoms() if a.residue.name not in SOLVENT]


def add_restraints(system, topology, positions):
    """Harmonic position restraints: CA 1.0, other protein heavy atoms 0.1 kcal/mol/A^2."""
    force = mm.CustomExternalForce("restraint_scale*k*periodicdistance(x,y,z,x0,y0,z0)^2")
    force.addGlobalParameter("restraint_scale", 1.0)
    force.addPerParticleParameter("k")
    for name in ("x0", "y0", "z0"):
        force.addPerParticleParameter(name)
    pos = positions.value_in_unit(unit.nanometer)
    for atom in topology.atoms():
        if atom.residue.name in SOLVENT or atom.element is None or atom.element.symbol == "H":
            continue
        k = (1.0 if atom.name == "CA" else 0.1) * KCAL_A2
        force.addParticle(atom.index, [k, *pos[atom.index]])
    system.addForce(force)


def make_simulation(topology, system, temperature, platform_name, seed=None):
    integrator = mm.LangevinMiddleIntegrator(temperature * unit.kelvin, 0.1 / unit.picosecond,
                                             4.0 * unit.femtosecond)
    integrator.setConstraintTolerance(1e-5)
    if seed is not None:
        integrator.setRandomNumberSeed(seed)
    if platform_name:
        platform = mm.Platform.getPlatformByName(platform_name)
        props = {"Precision": "mixed"} if platform_name in ("CUDA", "OpenCL") else {}
        return app.Simulation(topology, system, integrator, platform, props)
    return app.Simulation(topology, system, integrator)


def gpu_name():
    try:
        import subprocess
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def steps_for(ns):
    return int(round(ns * 1e6 / 4.0))    # 4 fs steps


def run_equil(args):
    t0 = time.time()
    topology, positions, make_system = load_build(args.build_dir)
    system = make_system()
    add_restraints(system, topology, positions)
    system.addForce(mm.MonteCarloBarostat(1.0 * unit.atmosphere, 300.0 * unit.kelvin, 25))
    sim = make_simulation(topology, system, 300.0, args.platform, seed=args.seed)
    sim.context.setPositions(positions)
    sim.minimizeEnergy(maxIterations=5000)
    sim.context.setVelocitiesToTemperature(300.0 * unit.kelvin, args.seed)
    sim.reporters.append(app.StateDataReporter(
        str(args.build_dir / "equil_log.csv"), 25000, time=True, temperature=True,
        potentialEnergy=True, volume=True, speed=True))

    half = steps_for(args.equil_ns / 2.0)
    md_start = time.time()
    sim.step(half)                                           # restrained
    sim.context.setParameter("restraint_scale", 0.0)
    sim.step(half)                                           # free
    md_seconds = time.time() - md_start

    state = sim.context.getState(getPositions=True, getVelocities=True)
    (args.build_dir / "equil_state.xml").write_text(mm.XmlSerializer.serialize(state))
    box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom)
    summary = dict(equil_ns=args.equil_ns, elapsed_s=time.time() - t0,
                   ns_per_day=args.equil_ns / md_seconds * 86400.0 if md_seconds > 0 else None,
                   final_box_A=float(box[0, 0]),
                   platform=sim.context.getPlatform().getName(), gpu=gpu_name(), seed=args.seed)
    (args.build_dir / "equil_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def run_prod(args, load=load_build):
    import mdtraj as md
    from mdtraj.formats import XTCTrajectoryFile

    t0 = time.time()
    equil = json.loads((args.build_dir / "equil_summary.json").read_text())
    deadline = t0 + args.budget_hours * 3600.0 - equil["elapsed_s"] - args.margin_min * 60.0

    topology, _, make_system = load(args.build_dir)
    sim = make_simulation(topology, make_system(), args.temperature, args.platform,
                          seed=args.seed)
    state = mm.XmlSerializer.deserialize((args.build_dir / "equil_state.xml").read_text())
    sim.context.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
    sim.context.setPositions(state.getPositions())
    sim.context.setVelocitiesToTemperature(args.temperature * unit.kelvin, args.seed)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    keep = protein_atoms(topology)
    md_top = md.Topology.from_openmm(topology).subset(keep)

    def frame():
        s = sim.context.getState(getPositions=True)
        xyz = s.getPositions(asNumpy=True).value_in_unit(unit.nanometer)[keep]
        box = s.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
        return np.asarray(xyz, dtype=np.float32), np.asarray(box, dtype=np.float32)

    xyz0, _ = frame()
    md.Trajectory(xyz0[None], md_top).save_pdb(str(out / "top.pdb"))

    steps_per_frame = int(round(args.frame_interval_ps * 1000.0 / 4.0))
    log_every = min(25000, steps_per_frame)
    sim.reporters.append(app.StateDataReporter(
        str(out / "log.csv"), log_every, time=True, temperature=True, potentialEnergy=True,
        speed=True))

    written, truncated, frame_seconds = 0, False, None
    md_start = time.time()
    with XTCTrajectoryFile(str(out / "traj.xtc"), "w") as xtc:
        def write(k):
            xyz, box = frame()
            xtc.write(xyz[None], time=np.array([k * args.frame_interval_ps], dtype=np.float32),
                      step=np.array([k * steps_per_frame], dtype=np.int32), box=box[None])
            xtc.flush()

        write(0)
        written = 1
        for k in range(1, args.n_frames):
            # Time one frame from the last one; before the first, from a short benchmark.
            if frame_seconds is None:
                bench = min(2500, steps_per_frame)
                tb = time.time()
                sim.step(bench)
                frame_seconds = (time.time() - tb) * steps_per_frame / bench
                remaining = steps_per_frame - bench
            else:
                remaining = steps_per_frame
            if time.time() + frame_seconds * remaining / steps_per_frame > deadline:
                truncated = True
                break
            tb = time.time()
            sim.step(remaining)
            if remaining == steps_per_frame:
                frame_seconds = time.time() - tb
            write(k)
            written += 1
    md_seconds = time.time() - md_start

    import pandas as pd
    log = pd.read_csv(out / "log.csv")
    temps = log[[c for c in log.columns if c.startswith("Temperature")][0]]
    ns = (written - 1) * args.frame_interval_ps / 1000.0
    summary = dict(temperature_K=args.temperature, seed=args.seed,
                   frames_requested=args.n_frames, frames_written=written,
                   frame_interval_ps=args.frame_interval_ps, simulated_ns=ns,
                   truncated_by_budget=truncated, budget_hours=args.budget_hours,
                   equil_elapsed_s=equil["elapsed_s"], prod_elapsed_s=time.time() - t0,
                   ns_per_day=ns / md_seconds * 86400.0 if ns > 0 else None,
                   mean_temperature_K=float(temps.mean()) if len(temps) else None,
                   n_protein_atoms=len(keep), n_atoms=topology.getNumAtoms(),
                   platform=sim.context.getPlatform().getName(), gpu=gpu_name())
    (out / "md_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["equil", "prod"])
    p.add_argument("--build-dir", type=Path, required=True)
    p.add_argument("--platform", default="CUDA", help="OpenMM platform; '' picks the fastest")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--equil-ns", type=float, default=20.0)
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--temperature", type=float)
    p.add_argument("--n-frames", type=int, help="frames to write, including the start frame")
    p.add_argument("--frame-interval-ps", type=float, default=1000.0)
    p.add_argument("--budget-hours", type=float, default=5.0)
    p.add_argument("--margin-min", type=float, default=10.0,
                   help="stop this many minutes before the budget ends")
    args = p.parse_args()
    if args.mode == "equil":
        run_equil(args)
    else:
        if args.out_dir is None or args.temperature is None or args.n_frames is None:
            p.error("prod needs --out-dir, --temperature and --n-frames")
        run_prod(args)


if __name__ == "__main__":
    main()

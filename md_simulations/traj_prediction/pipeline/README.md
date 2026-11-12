# Refold-and-resimulate pipeline

Tests whether a sequence from an inverse folding model reproduces the dynamics of
the mdCATH trajectory window it was designed from.

For each row of a manifest (trajectory, frame window, predicted sequence):

1. **prep** – cut the reference window out of the mdCATH shard (`prep_window.py`).
2. **fold** – fold the predicted and the native sequence with ColabFold/AF2 (`fold.sh`).
3. **MD** – build a solvated system, equilibrate it, and run production replicas
   with the mdCATH settings in OpenMM (`build_system.py`, `run_md.py`).
4. **analyze** – compare the new trajectories with the reference window
   (`analyze.py`, `compare.py`, `plots.py`).

The native sequence goes through the same steps as a control: it shows how much
difference the pipeline itself causes (AF2 start structure, OpenMM instead of ACEMD).

## Before the first run

1. Build and push the MD image (needs a machine with Docker):
   `docker build -t cmikulski4/traj-md:v1 . && docker push cmikulski4/traj-md:v1`
2. Check that GPU execute nodes reach the ColabFold server:
   `mkdir -p logs && condor_submit condor/connectivity_test.sub`, then read
   `logs/connectivity_test.out`.
3. Optional: store the AF2 weights once and pass them with `--af2-params`, so that
   jobs do not download 4 GB each time.

## Run

```bash
python3 build_dag.py manifest.csv --out runs      # see --help for the options
condor_submit_dag runs/pipeline.dag
```

Manifest columns: `traj_id` (`<domain>_<temperature>_<replica>`), `shard`,
`frame_start`, `frame_end` (0-based, both part of the window), `method`, `pred_seq`.
`manifest_example.csv` shows the format; its sequence is a placeholder.
The `shard` of each trajectory is in `../mdcath_320_test_max_rmsd.csv`.

A window of N frames is N − 1 ns, because mdCATH saves one frame per ns.

Results are in `runs/<traj_id>_<start>-<end>_<method>/analysis/`. Start with `report.md`.

## Time budget

`--budget-hours` (default 5) limits equilibration plus production of each
simulation. Production stops when the next frame would pass the limit, and the
analysis cuts every trajectory, the reference included, to the shortest one. The
report states when this happened. Use `--equil-ns` to shorten the equilibration
for tests.

## MD settings

From Mirarchi, Giorgino, De Fabritiis, *Sci. Data* 2024 (arXiv:2407.14794).

| Setting | mdCATH | Here |
|---|---|---|
| Engine | ACEMD | OpenMM (ACEMD's backend) |
| Builder | HTMD | HTMD |
| Start structure | CATH domain | AF2 model of the sequence |
| Protonation | pH 7 | pH 7 (`systemPrepare`) |
| Termini | acetylated, N-methylated | ACE, CT3 patches |
| Force field, water | CHARMM22*, TIP3P | same |
| Box | cubic, ≥ 9 Å padding | same |
| Ions | neutralised + 0.150 M NaCl | same |
| Time step | 4 fs, hydrogen mass 4 amu | same |
| Constraints | bonds to H, tolerance 1e-5 | same |
| Electrostatics | PME, 9 Å cutoff | same |
| van der Waals | 9 Å cutoff, switch from 7.5 Å | same |
| Thermostat | Langevin, 0.1 ps⁻¹ | `LangevinMiddleIntegrator`, 0.1 ps⁻¹ |
| Equilibration | 20 ns NPT, 1 atm, 300 K, Monte Carlo barostat; CA (1.0) and heavy-atom (0.1 kcal/mol/Å²) restraints for the first 10 ns | same |
| Production | NVT at 320–450 K, frame every 1 ns | NVT at the trajectory temperature, frame every 1 ns |
| Replicas | 5 per temperature from one equilibrated system | `--replicas` from one equilibrated system, new velocities |

Not in the paper, chosen here: energy minimisation (up to 5000 iterations) and
new Maxwell–Boltzmann velocities at the start of production.

## Metrics

See the docstring of `compare.py`. All metrics use backbone atoms only, because
the two sequences have different side chains and the shards store the backbone.

## Tests

`pytest tests/` checks the metrics on synthetic trajectories (needs mdtraj, scipy,
scikit-learn).

## Status

Tested locally: the metrics, `prep_window.py` and `analyze.py` on a synthetic
shard, the production loop and time-budget stop of `run_md.py` on a small CPU
system, and the submit files with `condor_submit -dry-run`.
Not yet run: the Docker image build, `build_system.py` (HTMD), ColabFold, and
GPU jobs. Run the smallest case first:

```bash
python3 build_dag.py manifest_example.csv --out runs_smoke --replicas 1 --native-replicas 1 --equil-ns 0.2
```

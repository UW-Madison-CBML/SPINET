# MD trajectories → DynamicMPNN

## Datasets, and why this model keeps the MD input

ATLAS and mdCATH are trained and evaluated **separately** — one run each, `--ds-name atlas`
or `--ds-name mdcath` — matching `scripts/train_residue_classifier.py`. Splits come from
`lib/dataset_splits.py`: ATLAS holds out two `cross_val` folds (one validation, one test);
mdCATH's topology split's `train`/`validation`/`test` rows are three disjoint splits.

`--val-fold` (default 0) selects the ATLAS validation fold and `--test-fold` (default 4) the
test fold; the other three train. Validation is scored every epoch, and the test split is
scored once after the last epoch -- there is no early stopping or best-epoch restore here, so
the tested weights are the trained weights and the test split never fed back into them.
`--max-length` (default `lib/dataset_splits.MAX_LENGTH`) drops the same long proteins
MapDiff and PiFold drop, and `--seed` (default `lib/dataset_splits.SEED`) seeds `random` and
`np.random` as well as torch -- the featurizer picks which *k* of each protein's saved
conformer pool to use with `random.sample`, once, at dataset-construction time, so leaving
those unseeded would change the validation set itself from run to run. Each run logs
`split_counts`, `val_split_ids` and `test_split_ids` to its W&B summary, as MapDiff and
PiFold do, so the three models' held-out sets can be diffed rather than assumed identical.

`comparisons/{MapDiff,PiFold}` are static-structure models, so they were retargeted onto each
protein's relaxed (deposited) PDB entry. DynamicMPNN is **not**: it consumes an ensemble of
conformers sampled from the trajectory, and that ensemble input is exactly the thing being
benchmarked, so Steps 1–3 below still read the MD store.

## Metrics logged (a la `scripts/train_residue_classifier.py`)

Every run additionally logs the *complete* metric set
`scripts/train_residue_classifier.py`'s `run_val` produces, under exactly the
same W&B keys, so the sheaf model and the three comparison models can be read
off one dashboard. That set is built by `lib.stats_utils.ResidueMetrics` (which
owns all of its conventions -- everything is accumulated per protein, and means
and stds are over proteins):

- `<split>_top{1,5,10}_acc_{mean,std}` -- per-protein recovery
- `<split>_perp_{mean,std}` and `epoch_<split>_loss` -- per-protein perplexity
- `<split>_<AA>_{f1,precision,recall}_{mean,std}` -- one triple per residue
  type, named with the uppercase three-letter code
- `<split>_aa_confusion_matrix` -- a `wandb.Image`
- `pred_seqs` (`<split>_pred_seqs` outside the val split) -- a `wandb.Table` of
  every argmax design, one row per protein, labelled by `pdb` (`<pdb>_<chain>`
  for ATLAS, the CATH domain id for mdCATH)
- `params` -- trainable parameter count

## ATLAS → DynamicMPNN: 5-fold CV eval script

Everywhere DynamicMPNN touches data, a "protein" is a small cluster of
discrete conformers rather than a trajectory.

- `PTFileDataset.__getitem__` loads one `{pdb_code}.pt` file, which is a PyG
  `Data` object with `cluster_members` (a list of conformer IDs), `pyg_dict`
  (id → per-conformer `Data`) and, optionally, the `tm_scores` /
  `tm_score_representatives` pair that drives TM-based *k* selection.
  (`src/dynamicmpnn/datamodules/pt_dataset.py`)
- `ProteinGraphFeaturiser.stack_conformations` **hard-errors if you give it
  more than 2 conformers** (`if len(confs_list) > 2: raise ValueError`), and
  `get_entries` explicitly *selects a pair* out of a larger cluster —
  uniformly at random, or by sampling low-TM-score (i.e. maximally
  dissimilar) pairs when `pair_tm_tau > 0`.
  (`src/dynamicmpnn/features/featurizer.py`)
- The single-chain variant (`ProteinGraphFeaturiserSingleChain`, used by the
  `k=2/3/5` checkpoints) generalizes this to picking *k* conformers via a
  greedy farthest-point-style scheme over TM-score dissimilarity
  (`compute_sequential_probabilities`).
- Even the standalone eval CLI (`dynamicmpnn-evaluate`, `eval/inputs.py`)
  takes exactly two named PDB *states* (e.g. `1bdt_1qtg` = apo/holo) and
  builds the same `cluster_members`/`pyg_dict` structure from them.

So DynamicMPNN was trained and is evaluated on **pairs (or small sets) of
structurally distinct states** — think apo/holo, open/closed — never on a
dense time series. If you feed it raw consecutive ATLAS frames:

Conformer selection here is the MD analogue of what DynamicMPNN already does
with static structures: pick the frames that best represent "distinct
conformational states" of the trajectory, scored with **DynamicMPNN's own
TM-score** (`dynamicmpnn.eval.scoring.compute_tm_score`) rather than an RMSD
proxy. Upstream fills its TM matrices from Foldseek all-vs-all alignments; that
alignment step is unnecessary here because every frame is the same chain, so the
residue correspondence is the identity and `compute_tm_score`'s Kabsch
superposition over CA atoms is the whole calculation.

`train.py` saves a pool of ~10 farthest-point-sampled frames per protein
together with their TM sub-matrix, as `tm_scores` /
`tm_score_representatives`. Those are the two fields
`ProteinGraphFeaturiserSingleChain.get_entries` looks for: with them present it
draws the *k* conformers it trains on by TM dissimilarity (its own
`compute_sequential_probabilities`); without them it silently falls back to
`random.sample`. An ensemble `.pt` built before this change has no `tm_scores`
and is rebuilt automatically.

## Sweeping *k*: how much does ensemble size actually buy?

`sweep_k.py` / `sweep_k.sub` train one model per *k* over a shared conformer
pool, so the recovery-vs-ensemble-size curve is measured rather than assumed.
`k` is the hyperparameter this whole comparison is about: MapDiff and PiFold
each see one static structure, DynamicMPNN sees an ensemble, and the sweep is
what says where that advantage saturates.

**There is no architectural ceiling on *k*.** The old limit of 10 was
`train.py`'s `--pool-size` default — how many frames preprocessing bothered to
save — not a property of the model. `DynamicMPNN.forward_single_chain` embeds
and encodes all *k* conformers before pooling them (node/edge activations are
`[N, k, D]` / `[E, k, D]` through the 8 encoder layers; the decoder runs on the
pooled, *k*-independent representation), so the real ceiling is VRAM, and the
cost is linear in *k* — a little worse than linear, because the edge topology
is the *union* of the *k* per-conformer knn_32 graphs, so `E` grows with *k*
too before saturating.

The sweep therefore budgets `k * batch_size` (`--conf-budget`, default 20 —
the load `train.sub`'s `--k 10 --batch-size 2` already carries) instead of
fixing the batch size, and makes up the difference with gradient accumulation
so every *k* still takes an optimizer step every `--proteins-per-step` (2)
proteins. Without that, the effective batch would shrink as *k* grew and the
sweep would be measuring batch size as much as ensemble size. The default grid
`2,3,4,5,6,8,10,12,16,20,24,32` peaks at 32 conformer-graphs resident, 1.6x
today's run.

Two things are shared across the grid rather than redone per *k*, because
neither depends on ensemble size and redoing them would add a second moving
variable: the conformer **pool** is built once at `--pool-size` = max(grid), so
every *k* draws from the same candidate frames. The seed is reset identically before
each point, so weight init and the *k*-of-pool draw are controlled too.

One trap worth naming: `get_entries` **silently duplicates** conformers when a
protein's pool is smaller than *k* (its `n < self.k` branch pads by random
resampling and the `len(confs_list) == k` assert still passes). Across a sweep
that reads as the curve flattening for a reason that has nothing to do with
dynamics, so `check_pool_sizes` promotes it to a hard failure before training
starts. A pool comes up short when a trajectory has fewer coarse-grained frames
than max(grid) — i.e. when `--course-grain` is too aggressive for it.

```bash
mkdir -p logs && condor_submit sweep_k.sub          # ATLAS, k = 2..32
```

Each *k* logs its own W&B run (`DynamicMPNN_atlas_ksweep_k<k>`), all under one
`group` so they overlay; a `_summary` run holds the `recovery_vs_k` table. The
same table is rewritten to `sweep_k_atlas.csv` after **every** completed point
and transferred back on eviction, so an interrupted sweep keeps what finished.

## Big-picture pipeline

```
pseudo-code

# ---------------------------------------------------------------
# Step 0: inputs already sitting in this directory
# ---------------------------------------------------------------
atlas_cross_val_index.csv   # pdb -> cross_val fold (0..4), pre-made 5-fold split
                             # (mdCATH: mdcath_320_0_topology_split.csv, domain -> split)
hdf5_data.py                 # <- the script we're fleshing out

# ---------------------------------------------------------------
# Step 1: pull frames out of the ATLAS hdf5 store
# ---------------------------------------------------------------
for pdb_code in atlas_index["pdb"]:
    traj = load_trajectory_from_hdf5(pdb_code)   # coords: [T, N_res, atoms, 3], seq
    align_frames_to_reference(traj)              # Kabsch/superpose so RMSD is meaningful

# ---------------------------------------------------------------
# Step 2: reduce each trajectory to the conformers DynamicMPNN wants
#         (this replaces "load pdbs directly" — see reasoning above)
# ---------------------------------------------------------------
def select_states(traj, pool_size=10):
    tm_matrix = pairwise_tm(traj.coords)                    # [T, T], DynamicMPNN's own
                                                             # compute_tm_score, CA only
    return farthest_point_sample(1 - tm_matrix, pool_size)   # greedy max-min TM
                                                             # dissimilarity; the featuriser
                                                             # then draws k of this pool from
                                                             # the same TM matrix

frame_ids = select_states(traj, k=2)   # match checkpoint (k=2 multi-chain, or k=2/3/5 single-chain)

# ---------------------------------------------------------------
# Step 3: repack selected frames into DynamicMPNN's .pt schema
#         (mirrors resolve_inputs() / _build_target_only_pyg_data()
#         in src/dynamicmpnn/eval/inputs.py)
# ---------------------------------------------------------------
def frame_to_pyg_data(traj, frame_idx):
    return Data(
        coords=...,        # [N_res, 3 (N/CA/C), 3]
        residue_type=...,  # int-encoded sequence
        residue_index=...,
        chains=...,
        homo_idx=...,      # single chain -> all zeros
        mask_seq=...,
    )

pyg_dict = {f"{pdb_code}_frame{i}": frame_to_pyg_data(traj, i) for i in frame_ids}
cluster_members = list(pyg_dict.keys())

torch.save(Data(pyg_dict=pyg_dict, cluster_members=cluster_members),
           processed_dir / f"{pdb_code}.pt")

# ---------------------------------------------------------------
# Step 4: 5-fold CV splits, reusing atlas_cross_val_index.csv directly
#         (don't re-derive folds — they're already assigned per pdb)
# ---------------------------------------------------------------
folds = atlas_index.groupby("cross_val")["pdb"].apply(list)   # {0: [...], ..., 4: [...]}

results = []
for held_out_fold in range(5):
    test_pdbs  = folds[held_out_fold]
    train_pdbs = concat(folds[f] for f in range(5) if f != held_out_fold)

    # Two options depending on what "evaluate" means here:
    #   (a) zero-shot: just run the pretrained checkpoint on test_pdbs,
    #       no training — folds only used to report variance across subsets
    #   (b) fine-tune: start from checkpoints/*.ckpt, continue training on
    #       train_pdbs, then test on the held-out fold (real CV)
    model = load_checkpoint("checkpoints/single_chain_k2.ckpt")
    if MODE == "finetune":
        model = train(model, dataset=PTFileDataset(train_pdbs, processed_dir))

    test_loader = build_dataloader(PTFileDataset(test_pdbs, processed_dir))
    fold_metrics = lightning_module.test_step_over(test_loader)  # e.g. seq recovery
    results.append(fold_metrics)

report(aggregate(results))   # mean +/- std across the 5 folds
```

## Why this reuses `PTFileDataset` / the Lightning module, not the eval CLI

`dynamicmpnn-evaluate` (`eval/pipeline.py`) is built for *one target at a
time*, optionally running AlphaFold3 self-consistency — not batched metric
computation over hundreds of ATLAS trajectories. The training-side path
(`PTFileDataset` + `MultiConfProteinDataModule` + `lightning_module.test_step`)
is the bulk-evaluation path DynamicMPNN itself uses for its own val/test sets,
and it's the one that scales to "run this over every fold." Build `.pt` files
in that schema and you get the existing dataloader, batching, and metric
code for free — you only need to write the ATLAS → `.pt` conversion
(Steps 1–3 above) and the fold loop (Step 4).

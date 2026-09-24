# MD trajectories → DynamicMPNN

## Datasets, and why this model keeps the MD input

ATLAS and mdCATH are trained and evaluated **separately** — one run each, `--ds-name atlas`
or `--ds-name mdcath` — matching `scripts/train_residue_classifier.py`. Splits come from
`lib/dataset_splits.py`: ATLAS holds out two `cross_val` folds (one validation, one test);
mdCATH's topology split's `train`/`validation`/`test` rows are three disjoint splits.

`--val-fold` (default 0) selects the ATLAS validation fold and `--test-fold` (default 4) the
test fold; the other three train. Validation is scored every epoch and training early-stops
on val perplexity (`--early-stopping-patience`, default 10); the test split is scored once,
with the lowest-val-perplexity epoch's weights (also saved as `weights/<run>_<id>_best.ckpt`),
so the test split never fed back into the weights or the choice of epoch. MapDiff and PiFold
select the same way, so every comparison row shares one protocol. Training proteins are
re-featurised on every draw (`ResamplingDataset`): a fresh draw of *k* conformers from the
pool plus the featuriser's train-time coordinate noise, as upstream's `in_memory: False`
trained. Val/test are featurised once, without noise.
`--max-length` (default `lib/dataset_splits.MAX_LENGTH`) drops the same long proteins
MapDiff and PiFold drop, and `--seed` (default `lib/dataset_splits.SEED`) seeds `random` and
`np.random` as well as torch -- the featurizer picks which *k* of each protein's saved
conformer pool to use with `random.sample`, and for val/test that happens once, at
dataset-construction time, so leaving those unseeded would change the validation set itself
from run to run. Each run logs
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

## Zero-shot: scoring the *released* checkpoints

`eval_ckpt.py` / `eval_ckpt.sh` / `eval_ckpt.sub` run upstream's published
weights over our held-out splits with no training at all:

```
condor_submit eval_ckpt.sub                        # ATLAS, single_chain_k2.ckpt
condor_submit eval_ckpt.sub DS_NAME=mdcath \
    DS_H5='$(ResearchDrive)/mdcath_spinet_320_0.h5' \
    DS_CSV='$(ResearchDrive)/mdcath_320_0_topology_split.csv'
```

Same `.pt` conformer ensembles, same splits, same `ResidueMetrics` keys as
`train.py` — it imports the conversion and the scoring loop from it — so the
zero-shot row drops straight into the table next to the from-scratch row. Both
held-out splits are scored in one pass (nothing here is trained on, so there is
no reason to hold `test` back for the end).

Two things are not obvious and are worth knowing before reading the numbers:

* **The released checkpoints are not the architecture `train.py` builds.** Their
  own `hyper_parameters` record the legacy pair model — `node_in_dim (26, 2)` /
  `edge_in_dim (17, 1)`, 4 encoder + 4 decoder layers, `representation: ca`, no
  `sequence` node feature and no `edge_type` edge feature — against
  `AR1_single_chain.yaml`'s 27/18, 8+4 and `pooling_strategy: single_chain_k`.
  `config_from_ckpt` reads the config out of the checkpoint and retargets its
  `dynamicprot_single.src.*` class paths onto the vendored `dynamicmpnn.*` ones,
  so k2/k3/k5 all load without a hand-written yaml per checkpoint.
* **All three are *pair* models, whatever the file name says.** The featuriser
  they were trained with (`ProteinGraphFeaturiser`, not
  `…SingleChain`) is hard-coded to two conformations, so `single_chain_k5.ckpt`
  sees 2 conformers here, not 5 — the run logs `k: 2` accordingly. It also keys
  `pyg_dict` by `member.split('_')[0].upper()`, i.e. codnas' `<PDB>_<chain>`,
  where our pools key by `<pdb>_frame<t>`; fed ours unadapted it would resolve
  both members to the same frame and score a "2-conformer" model on one
  structure. `LegacyPairAdapter` picks the pair (by default the most TM-dissimilar
  one in the pool, deterministically) and re-presents it under the naming and the
  per-conformer fields that featuriser expects, leaving the actual feature
  computation to upstream's own code.

A third wrinkle is internal: in the pair path the encoder graph carries two nodes
per residue while the decoder — and therefore the logits — carries one, pooled
over the pair, so `batch.batch` is twice as long as `run_validation` expects.
`DecoderNodeBatching` rewrites it from `num_decoder_nodes` after the forward
pass; without it the per-protein slicing would be silently wrong.

Expect ~36% top-1 on well-converted data: that is what `single_chain_k2.ckpt`
scored on upstream's own `val_pt_single_chain` files, and it is the check that
caught the nanometre/Angstrom bug documented at `train.py`'s `NM_TO_ANGSTROM`
(7.3% recovery, loss 116, before the conversion). A pretrained model scoring near
chance on our `.pt` files is evidence about *our features*, not about the model.

## Fine-tuning the released checkpoints

`finetune.py` / `finetune.sh` / `finetune.sub` start from upstream's weights and keep
training on our training split. This row sits between from-scratch and zero-shot:

```
condor_submit finetune.sub                                   # ATLAS, single_chain_k2.ckpt, lr 1e-3, early-stopped
condor_submit finetune.sub CKPT=DynamicMPNN/checkpoints/single_chain_k5.ckpt LR=1e-4
```

The model, featuriser, pair adapter and batching fix all come from `eval_ckpt.py`, so
everything in the zero-shot section above applies here too: it is the legacy pair model, and
it sees k = 2. What's new is on the training side:

* **Training pairs are re-featurised on every draw** (`ResamplingDataset`, as in `train.py`):
  a fresh random pair from the pool, plus the featuriser's train-time coordinate noise, each
  time. Upstream trained these weights with `in_memory: False`, which works the same way.
  `--train-pair-selection tm_min` always uses the most dissimilar pair instead.
* **The paper's optimisation regime**: Adam at lr 1e-3 and 32 proteins per optimizer step
  (2 per forward pass, gradients accumulated over 16; `--effective-batch-size`).
* **Model selection as in `train.py`**: early stopping on val perplexity (patience 10, at most
  `--epochs` 200), best val epoch restored for test. The zero-shot epoch -1 is a candidate,
  so if no update beats it, test scores the released weights.
* **Val/test are featurised exactly as `eval_ckpt.py` does them** (`tm_min`, no noise). Val is
  also scored once before the first update, logged at `epoch = -1`, so the fine-tuning curve
  starts from the zero-shot number on the same proteins.
* Saved weights carry upstream's `hyper_parameters['cfg']` layout, so
  `eval_ckpt.py --ckpt weights/<run>.ckpt` rescores a fine-tuned run, and
  `finetune.py --ckpt weights/<run>.ckpt` continues one.

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
same table is rewritten to `--results-csv` after **every** completed point
and transferred back on eviction, so an interrupted sweep keeps what finished.

**Running the grid in chunks.** Twelve points x 8 epochs is a long single job on a
shared machine, and a sweep that is evicted or OOMs in the tail is resumed by
submitting the rest of the grid rather than the whole thing again. Two invariants:

```bash
condor_submit sweep_k.sub K_GRID=2,3,4,5,6,8,10,12,16,20 POOL_SIZE=32 RESULTS_CSV=sweep_k_atlas_k2_20.csv
condor_submit sweep_k.sub K_GRID=24,32                   POOL_SIZE=32 RESULTS_CSV=sweep_k_atlas_k24_32.csv
python merge_sweep_csvs.py sweep_k_atlas.csv sweep_k_atlas_k2_20.csv sweep_k_atlas_k24_32.csv
```

- `POOL_SIZE` stays at the top of the **full** grid in every chunk, so each *k* draws
  from the same candidate frames. A chunk rebuilds the pool from scratch in its own
  job scratch, which is safe because `farthest_point_sample` is a pure argmax with no
  RNG: same trajectory + same `--course-grain` + same `--pool-size` gives the same pool,
  and the greedy selection means a pool of 32 extends the pool of 20 rather than
  replacing it.
- `RESULTS_CSV` is **unique per chunk**. `write_results_csv` opens with `w`, so two
  chunks sharing a filename means the second job's `transfer_output_files` overwrites
  the first job's numbers on the way home. `merge_sweep_csvs.py` stitches the chunks
  back into one k-sorted table (union of columns; a *k* present in two chunks is an
  error, not a silent overwrite).

Each chunk's `_summary` W&B run holds only that chunk's points, so the merged CSV --
not W&B -- is the source of truth for the full recovery-vs-*k* curve.

One caveat when reading the curve across chunks: `batch_size_for_k` floors `batch_size`
at 1 once `k >= conf_budget` (20), so k = 12, 16, 20, 24 and 32 all run at
`batch_size` 1 x `grad_accum` 2 while k <= 10 runs at 2 x 1. The optimizer batch is
2 proteins throughout, but the k <= 10 and k >= 12 halves of the grid differ by how
that batch is split -- worth stating explicitly rather than reading the k=10 -> k=12
step as an ensemble-size effect.

## mdCATH: two things that differ from ATLAS

**Coordinate units are not the same in the two stores, and the code no longer assumes they
are.** Measured median consecutive CA-CA distance over 60 trajectories per store: ATLAS
0.3835 (nanometres, straight out of `traj.xyz`), mdCATH 3.8357 (already Angstroms).
`build_ensemble` used to multiply both by `NM_TO_ANGSTROM`, which for mdCATH inflates every
coordinate 10x. That is silent in exactly the way the nm-vs-A note at the top of `train.py`
describes, only in reverse: knn topology and the angle features are scale-invariant, but every
`edge_distance` / `rbf_16` channel saturates, and TM-score's `d0` is an Angstrom quantity, so
every frame reads as maximally dissimilar. Measured min-pairwise-TM over 40 mdCATH domains is
**0.046 scaled vs 0.682 unscaled** -- i.e. under the bug, farthest-point selection draws its
pool out of numerical noise and the `is_flat` warning can never fire. `detect_coord_scale`
now reads the factor off the store's own CA-CA geometry and raises rather than guessing if it
matches neither convention.

Any mdCATH `.pt` files built before this fix are wrong, and `needs_rebuild` will not catch
them (it only checks that `tm_scores` exists). A condor run starts from empty job scratch so
it rebuilds anyway; pass `--force-rebuild` if you are reusing a persistent `--processed-dir`.
ATLAS is unaffected -- it still resolves to 10.0, so the completed k = 2..20 sweep stands.

**`--course-grain` is now per dataset**, because the trajectories are very different lengths:

| store  | T (frames)              | cg | pool per protein     | supports k=32? |
|--------|-------------------------|----|----------------------|----------------|
| ATLAS  | 1001, every trajectory  | 25 | 41, every trajectory | yes            |
| mdCATH | 130-501 (median 500)    | 25 | 6-21 (median 20)     | **no**         |
| mdCATH | 130-501 (median 500)    | 4  | 33-126 (median 125)  | yes            |

At cg=25 *no* mdCATH protein can support k > 21 and 2074 of the 5327 split members cannot
support k = 20 either, which is what `check_pool_sizes` catches. cg=4 is the largest stride
that keeps the shortest trajectory (130 frames) at >= 32. `DEFAULT_COURSE_GRAIN` in `train.py`
holds both values; ATLAS deliberately stays at 25 so its finished sweep stays reproducible.

The cost is preprocessing. `compute_pairwise_tm` is a serial O(n_coarse^2) Kabsch loop:
measured 0.21 s/protein at cg=25 and 1.38 s/protein at cg=4, so building the mdCATH pool is
**~2 h** for 5327 proteins versus ~19 min at cg=25. It is paid once per pool directory, but
job scratch does not persist, so each chunked mdCATH submission pays it again -- worth
preferring one long job over several chunks here, the opposite of the ATLAS advice above.

```bash
condor_submit sweep_k.sub DS_NAME=mdcath \
    DS_H5='$(ResearchDrive)/mdcath_spinet_320_0.h5' \
    DS_CSV='$(ResearchDrive)/mdcath_320_0_topology_split.csv' \
    RESULTS_CSV=sweep_k_mdcath.csv
```

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

# ATLAS → DynamicMPNN: 5-fold CV eval script

Everywhere DynamicMPNN touches data, a "protein" is a small cluster of
discrete conformers rather than a trajectory.

- `PTFileDataset.__getitem__` loads one `{pdb_code}.pt` file, which is a PyG
  `Data` object with exactly two fields: `cluster_members` (a list of
  conformer IDs) and `pyg_dict` (id → per-conformer `Data`).
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

The RMSD-pair approach is also just the MD analogue of what DynamicMPNN
already does with static structures: pick the two (or k) frames that best
represent "distinct conformational states" of the trajectory.

## Big-picture pipeline

```
pseudo-code

# ---------------------------------------------------------------
# Step 0: inputs already sitting in this directory
# ---------------------------------------------------------------
atlas_cross_val_index.csv   # pdb -> cross_val fold (0..4), pre-made 5-fold split
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
def select_states(traj, k=2):
    rmsd_matrix = pairwise_ca_rmsd(traj.coords)      # [T, T]
    if k == 2:
        i, j = argmax(rmsd_matrix)                   # the single most-dissimilar pair
        return [i, j]
    else:
        return farthest_point_sample(rmsd_matrix, k)  # greedy max-min RMSD, same
                                                       # spirit as DynamicMPNN's own
                                                       # TM-dissimilarity k-selection

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

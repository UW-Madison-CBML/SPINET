"""The train/val(/test) partition of each shared dataset, in one place.

`scripts/train_residue_classifier.py` trains and evaluates the sheaf model on ATLAS and
mdCATH separately; every comparison model in `comparisons/` now does the same, and all of
them have to cut the splits identically or the numbers aren't comparable. The two datasets
are indexed differently:

* **ATLAS** -- `atlas_cross_val_index.csv`, column `pdb` (ids like ``1a0a_A``), column
  `cross_val` giving a pre-assigned 5-fold split. One fold (0 by default) is held out; the
  other four train. There is no separate test set, so the held-out fold doubles as the test
  split for models (MapDiff, PiFold) that insist on three.
* **mdCATH** -- `mdcath_320_0_topology_split.csv`, column `domain` (CATH domain ids like
  ``1a0aA02``), column `split` with values train/test/validation. `train` and `test` are a
  *topology-based* split of the training pool, so both train; **evaluation is on
  `validation` only**, matching train_residue_classifier.py.

The module also owns the handful of *non-split* knobs that have to agree across models for
the comparison to stay apples-to-apples -- which cross-validation fold is held out, how long
a protein may be before it is dropped, and the RNG seed -- so that changing one of them
changes it everywhere rather than in one model's argparse default.
"""
import pandas as pd

DATASETS = ("atlas", "mdcath")

# Which ATLAS cross_val fold is held out. scripts/train_residue_classifier.py hardcodes fold
# 0 (it reads `index["cross_val"] == 0`), so every comparison model defaults to the same one.
# mdCATH ignores this -- its split column is categorical, not a fold index.
DEFAULT_VAL_FOLD = 0

# Proteins longer than this (in residues) are dropped, by every model, from every split.
# The binding constraint is MapDiff's IPA node encoder, whose fixed positional-encoding table
# is sized by `model.ipa_pe_max_len` (conf/model/egnn.yaml) -- but the cutoff has to be shared
# or the models are scored on different proteins. Keep this and `ipa_pe_max_len` in step.
MAX_LENGTH = 1200

# One seed for every comparison run, so that anything still drawing from an RNG (DynamicMPNN's
# k-of-pool conformer subsampling, weight init, batch order) is reproducible run to run.
SEED = 42

# Default file names as staged by the submit files (see scripts/train_residue_classifier.sub
# and scripts/train_residue_classifier_mdcath.sub).
DEFAULT_INDEX_CSV = {
    "atlas": "atlas_cross_val_index.csv",
    "mdcath": "mdcath_320_0_topology_split.csv",
}
DEFAULT_H5 = {
    "atlas": "atlas_data.h5",
    "mdcath": "mdcath_spinet_320_0.h5",
}
# `id_column` names the column holding the protein id; ATLAS trajectories are truncated to
# 200 frames everywhere in this repo, mdCATH's are used whole.
ID_COLUMN = {"atlas": "pdb", "mdcath": "domain"}


def default_index_csv(ds_name):
    return DEFAULT_INDEX_CSV[_check(ds_name)]


def default_h5(ds_name):
    return DEFAULT_H5[_check(ds_name)]


def id_column(ds_name):
    return ID_COLUMN[_check(ds_name)]


def _check(ds_name):
    ds_name = ds_name.lower()
    if ds_name not in DATASETS:
        raise ValueError("unknown dataset {!r}, expected one of {}".format(ds_name, DATASETS))
    return ds_name


def get_splits(ds_name, index_csv=None, val_fold=DEFAULT_VAL_FOLD):
    """``(train_ids, val_ids)`` for ``ds_name``.

    ``val_ids`` is the *only* evaluation split for both datasets -- for ATLAS it is the
    held-out cross-validation fold, for mdCATH the `validation` rows. Callers that need a
    third "test" split should reuse ``val_ids``; neither dataset has a further held-out set.
    """
    ds_name = _check(ds_name)
    index_df = pd.read_csv(index_csv or default_index_csv(ds_name))

    if ds_name == "atlas":
        val_mask = index_df["cross_val"] == val_fold
        return index_df.loc[~val_mask, "pdb"].tolist(), index_df.loc[val_mask, "pdb"].tolist()

    # mdCATH: train on the topology split's train+test pools, evaluate on validation only.
    train_ids = index_df.loc[index_df["split"].isin(["train", "test"]), "domain"].tolist()
    val_ids = index_df.loc[index_df["split"] == "validation", "domain"].tolist()
    return train_ids, val_ids


def all_ids(ds_name, index_csv=None):
    """Every protein id in the dataset's index, in file order."""
    ds_name = _check(ds_name)
    index_df = pd.read_csv(index_csv or default_index_csv(ds_name))
    return index_df[id_column(ds_name)].tolist()

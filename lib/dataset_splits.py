"""The train/val(/test) partition of each shared dataset, in one place.

`scripts/train_residue_classifier.py` trains and evaluates the sheaf model on ATLAS and
mdCATH separately; every comparison model in `comparisons/` now does the same, and all of
them have to cut the splits identically or the numbers aren't comparable. The two datasets
are indexed differently:

* **ATLAS** -- `atlas_cross_val_index.csv`, column `pdb` (ids like ``1a0a_A``), column
  `cross_val` giving a pre-assigned 5-fold split. Fold `val_fold` (0 by default) is the
  validation split and fold `test_fold` (4) is the held-out **test** split; the remaining
  three folds train. The test fold is never trained on.
* **mdCATH** -- `mdcath_320_0_topology_split.csv`, column `domain` (CATH domain ids like
  ``1a0aA02``), column `split` with values train/test/validation. Each value is its own
  split: `train` trains, `validation` is the per-epoch evaluation split, and `test` is the
  held-out test split (it used to be lumped in with `train`).

The module also owns the handful of *non-split* knobs that have to agree across models for
the comparison to stay apples-to-apples -- which cross-validation folds are held out, how
long a protein may be before it is dropped, and the RNG seed -- so that changing one of them
changes it everywhere rather than in one model's argparse default.

The file names below are only fallbacks. mdCATH comes in several builds -- one
`mdcath_spinet_<temperature>_<replica>.h5` store per simulation temperature (320 K, 450 K, ...),
each with its own split csv -- so a job picks one by setting `$DS_H5_FILE` / `$DS_CSV_FILE` to
the names of the files it staged; every comparison's `.sub` derives them from its `DS_H5` /
`DS_CSV` paths. `dataset_tag` turns that choice into a short label (``mdcath450``) for W&B run
names and cache directories, so runs and featurizations of different builds never collide.
"""
import os
import re

import pandas as pd

DATASETS = ("atlas", "mdcath")

DEFAULT_VAL_FOLD = 0

DEFAULT_TEST_FOLD = 4

MAX_LENGTH = 3000

SEED = 42

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
    return os.environ.get("DS_CSV_FILE") or DEFAULT_INDEX_CSV[_check(ds_name)]


def default_h5(ds_name):
    return os.environ.get("DS_H5_FILE") or DEFAULT_H5[_check(ds_name)]


def dataset_tag(ds_name, h5=None):
    """Short label naming the dataset *build*, e.g. ``atlas``, ``mdcath320``, ``mdcath450``.

    Read off the trajectory store's file name (``h5``, default `default_h5`): an mdCATH store
    named ``mdcath_spinet_<temperature>_<replica>.h5`` gives ``mdcath<temperature>``, plus
    ``_r<replica>`` for any replica but 0. Anything else is just the dataset name.
    """
    ds_name = _check(ds_name)
    name = os.path.basename(str(h5 or default_h5(ds_name)))
    match = re.fullmatch(r"mdcath_spinet_(\d+)_(\d+)\.h5", name)
    if ds_name != "mdcath" or match is None:
        return ds_name
    temperature, replica = match.groups()
    return "mdcath{}".format(temperature) + ("" if replica == "0" else "_r{}".format(replica))


def id_column(ds_name):
    return ID_COLUMN[_check(ds_name)]


def _check(ds_name):
    ds_name = ds_name.lower()
    if ds_name not in DATASETS:
        raise ValueError("unknown dataset {!r}, expected one of {}".format(ds_name, DATASETS))
    return ds_name


def get_splits(ds_name, index_csv=None, val_fold=DEFAULT_VAL_FOLD, test_fold=DEFAULT_TEST_FOLD):
    """``(train_ids, val_ids, test_ids)`` for ``ds_name``.

    Three disjoint splits for both datasets: ``val_ids`` is the split scored every epoch (and
    the one model selection runs on), ``test_ids`` is held out entirely and scored once with
    the selected model. Neither appears in ``train_ids``.

    ``val_fold``/``test_fold`` are ATLAS-only -- mdCATH's split column is categorical, so its
    three splits come straight from the `split` values train/validation/test.
    """
    ds_name = _check(ds_name)
    index_df = pd.read_csv(index_csv or default_index_csv(ds_name))

    if ds_name == "atlas":
        if val_fold == test_fold:
            raise ValueError(
                "val_fold and test_fold are both {} -- the validation and test splits would be "
                "the same proteins".format(val_fold)
            )
        val_mask = index_df["cross_val"] == val_fold
        test_mask = index_df["cross_val"] == test_fold
        return (index_df.loc[~(val_mask | test_mask), "pdb"].tolist(),
                index_df.loc[val_mask, "pdb"].tolist(),
                index_df.loc[test_mask, "pdb"].tolist())

    train_ids = index_df.loc[index_df["split"] == "train", "domain"].tolist()
    val_ids = index_df.loc[index_df["split"] == "validation", "domain"].tolist()
    test_ids = index_df.loc[index_df["split"] == "test", "domain"].tolist()
    return train_ids, val_ids, test_ids


def all_ids(ds_name, index_csv=None):
    """Every protein id in the dataset's index, in file order."""
    ds_name = _check(ds_name)
    index_df = pd.read_csv(index_csv or default_index_csv(ds_name))
    return index_df[id_column(ds_name)].tolist()

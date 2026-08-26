# make dirs for train/validation/test split for SurfFold

import shutil
import os
import requests
import numpy as np
import pandas as pd
from zipfile import ZipFile
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import traceback
from itertools import product

val_ratio = 0.15
test_ratio = 0.15

df = pd.read_csv(os.path.join("md_data", "atlas_index.csv"))

pdb_ids = df["pdb_id"].unique() # deterministic
num_pdbs = len(pdb_ids)

val_cutoff = int(val_ratio * num_pdbs)
test_cutoff = val_cutoff + int(test_ratio * num_pdbs)

val_pdbs = pdb_ids[:val_cutoff]
test_pdbs = pdb_ids[val_cutoff:test_cutoff]
train_pdbs = pdb_ids[test_cutoff:]

# sort into the dirs
for split, pdbs in zip(['validation', 'test', 'train'], [val_pdbs, test_pdbs, train_pdbs]):
    for pdb in pdbs:
        # save "pdb".pdb to corresponding folder train, validation, or test in surffold_data/
        os.rename(f"md_data/{pdb}.pdb", f"surffold_data/{split}/{pdb}.pdb")


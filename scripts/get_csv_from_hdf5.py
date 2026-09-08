import h5py
import pandas as pd, numpy as np
import os


if __name__ == "__main__":
    np_rng = np.random.default_rng(seed=42)
    h5_path = os.path.abspath("atlas_data.h5")
    with h5py.File(h5_path, "r") as f:
        all_pdbs = list(f.keys())
    np_rng.shuffle(all_pdbs)

    n = np.floor( (5 / len(all_pdbs)) * np.arange(len(all_pdbs))).astype(int)
    indices = np_rng.integers(1000-1, size=len(all_pdbs))
    df = pd.DataFrame({"pdb":all_pdbs, "cross_val":n, "random_indices":indices})
    df.to_csv(os.path.abspath("atlas_cross_val_index.csv"))

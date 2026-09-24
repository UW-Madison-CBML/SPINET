import numpy as np
from itertools import product
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os

LAPLACIAN_TYPES = ["first_sheaf", "last_sheaf", "shuffled_first_sheaf", "shuffled_last_sheaf", "graph"]
def get_top_eigs(h5_file, group_name):
    mat_test1 = h5_file[group_name]["first_sheaf"][:]
    mat_test2 = h5_file[group_name]["graph"][:]
    laplacians = {ds_name: h5_file[group_name][ds_name][:] for ds_name in LAPLACIAN_TYPES}
    for laplacian in laplacians.values():
        assert np.allclose(laplacian, laplacian.T)
    eigvals = {key: np.linalg.eigvalsh(item).real for key, item in laplacians.items()}

    # we just need the real components here, sheaf laplacian is real positive semi-definite
    #eigvals = {name: item.eigenvalues.real for name, item in eigs.items()}
    #eigvecs = {name: item.eigenvectors.real for name, item in eigs.items()}
    indices = {name: np.argsort(item) for name, item in eigvals.items()}
    df = pd.DataFrame()
    for ds_name,n in product(LAPLACIAN_TYPES, range(num_top)):
        df[f"{ds_name}_top_{n}"] = [eigvals[ds_name][indices[ds_name][-(n+1)]]]
        df[f"{ds_name}_bottom_{n}"] = [eigvals[ds_name][indices[ds_name][n]]]
    df["pdb_id"] = group_name
    return df

if __name__ == "__main__":
    eig_dfs = []
    num_top = 5
    h5_path = sys.argv[1] # "mdcath_laplacians.h5" or atlas_...
    with h5py.File(h5_path, "r", libver="latest", swmr=True) as h5_file:
        max_workers = os.cpu_count()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(get_top_eigs, h5_file, group_name): group_name for group_name in h5_file.keys()}

            kwargs = {"total": len(futures), "desc": "Processing PDB jobs", "unit": "job"}
            for future in tqdm(as_completed(futures), **kwargs):
                try:
                    df = future.result()
                    eig_dfs.append(df)
                except AssertionError as e:
                    print(f"Worker generated an assertion exception: {e}")
                except Exception as e:
                    print(f"Worker generated an exception: {e}")

    out_df = pd.concat(eig_dfs, axis=0, ignore_index=True)
    out_df.set_index("pdb_id", inplace=True)
    out_df.to_csv("eigval_df.csv")



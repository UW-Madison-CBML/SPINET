import numpy as np
import h5py 
import matplotlib.pyplot as plt
import pandas as pd



LAPLACIAN_TYPES = ["first_sheaf", "last_sheaf", "shuffled_first_sheaf", "shuffled_last_sheaf", "graph"]

if __name__ == "__main__":
    eig_dfs = [] 
    num_top = 5
    h5_path = sys.argv[1] # "mdcath_laplacians.h5" or atlas_...
    with h5py.File(h5_path, "r") as h5_file:
        for group_name in h5_file.keys():
            eigs = {ds_name: np.linalg.eig(h5_file[group_name][ds_name]) for ds_name in LAPLACIAN_TYPES}

            # we just need the real components here, sheaf laplacian is real positive semi-definite
            eigvals = {name: item.eigenvalues.real for name, item in eigs.items()}
            eigvecs = {name: item.eigenvectors.real for name, item in eigs.items()}
            indices = {name: np.argsort(item) for name, item in eigvals.items()}
            top_dict = {f"top_{n}":[] for n in range(num_top)}                 
            bottom_dict = {f"bottom_{n}":[] for n in range(num_top)}
            for ds_name in LAPLACIAN_TYPES:
                for n in range(num_top):
                    top_dict[f"top_{n}"].append(eigvals[ds_name][indices[ds_name][-(n+1)]])
                    bottom_dict[f"bottom_{n}"].append(eigvals[ds_name][indices[ds_name][n]])
            df = pd.DataFrame(top_dict | bottom_dict, index=LAPLACIAN_TYPES)
            df["pdb_id"] = group_name
            eig_dfs.append(df)
    out_df = pd.concat(eig_dfs, axis=0, ignore_index=True)
    out_df.to_csv("eigval_df")





        

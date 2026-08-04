import requests 
import pandas as pd
import numpy as np
import threading
import os
from zipfile import ZipFile
import mdtraj
def process_traj(traj): #, features):
    top = traj.topology
    residues = list(top.residues)
    n_frames = traj.n_frames
    n_res = top.n_residues

    centroids = np.empty((n_frames, n_res, 3))
    ca_idx = np.array([top.select(f"resid {r.index} and name CA")[0] for r in residues]) #c alpha
    n_idx  = np.array([top.select(f"resid {r.index} and name N")[0]  for r in residues]) #n-h 
    c_idx  = np.array([top.select(f"resid {r.index} and name C")[0]  for r in residues]) #c-o

    for i, res in enumerate(residues):
        heavy_idx = top.select(f"resid {res.index} and not element H")
        sub = traj.atom_slice(heavy_idx)           # (n_frames, n_heavy_atoms_i, 3)
        masses = np.array([a.element.mass for a in sub.topology.atoms])
        centroids[:, i, :] = (sub.xyz * masses[None, :, None]).sum(axis=1) / masses.sum()
     
    differences = np.diff(centroids, axis=0)
    features = np.concatenate([differences, centroids[1:]], axis=2)

    timesteps = np.broadcast_to(np.arange(features.shape[0])[:, None], features.shape[:2])



    targets = residues * features.shape[0]

    features = features.reshape(-1, 6) # res * time, features
    timesteps = timesteps.flatten()
    
    features_df = pd.DataFrame(features, columns = ["x","y","z","dx","dy","dz"]) 
    metadata_df = pd.DataFrame(timesteps, columns=["timestep"], index = features_df.index) # this will be a groupby column
    targets_df = pd.DataFrame(targets, columns=["residue"], index = features_df.index)

    df = pd.concat([metadata_df, features_df, targets_df], axis=1)
    return df
    

def download_and_process_file(url, pdb_id):
    zip_file = os.path.join("md_data",f"{pdb_id}.zip")
    folder = os.path.join("md_data",f"{pdb_id}")
    try:
        with requests.get(url, stream=True) as r: # stream=True is very important
            r.raise_for_status()
            with open(zip_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)
        
        os.makedirs(folder)
        with ZipFile(zip_file, 'r') as zObject:
            zObject.extractall(path=folder)
        traj_ids = [f"{pdb_id}_R{i}" for i in range(1,4)]
        dfs = []
        for eye_d in traj_ids:
            traj = mdtraj.load_xtc(os.path.join(folder, f"{eye_d}.xtc"), top=os.path.join(folder, f"{pdb_id}.pdb"))
            df = process_traj(traj)
            df["traj_id"] = eye_d
        
            dfs.append(df)
        trajectories_df = pd.concat(dfs, axis=0, ignore_index=True) 
        trajectories_df.to_csv(os.path.join(folder, f"{pdb_id}.csv"))
    except requests.exceptions.HTTPError as errh:
        print(f"failed to read {url}") 


def main():
    threads = {}
    base_url = "https://www.dsimb.inserm.fr/ATLAS/api" 
    atlas_df = pd.read_csv(os.path.abspath("atlas.csv"))
    # for whatever reason the format for the api is slightly different
    atlas_df["pdb"] = atlas_df["pdb"].map(lambda x: x[:4] + "_" + x[-1])
    md_urls = [(base_url + f"/ATLAS/analysis/{pdb}", pdb) for pdb in atlas_df["pdb"].to_list()]

    for url, pdb in md_urls:
        print(url)
        threads[url] = threading.Thread(target=download_and_process_file, args=(url,pdb))
    for i in threads.keys():
        threads[i].start()
    for i in threads.keys():
        threads[i].join()
     
        

if __name__ == "__main__":
    main()

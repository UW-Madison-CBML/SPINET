"""
import requests 
import pandas as pd
import numpy as np
import threading
import os
from zipfile import ZipFile
import mdtraj
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tqdm import tqdm


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
    # first look at angles and distances
    sequence_differences = np.diff(centroids, axis=1)
    bond_len = np.linalg.norm(sequence_differences, axis=2)
    bond_len = np.pad(bond_len, ((0,0),(0,1)), mode="edge")
 
    v1, v2 = centroids[:,1:-1] - centroids[:,:-2], centroids[:,2:] - centroids[:,1:-1]
    cosang = np.einsum('tij,tij->ti', v1, v2) / (
        np.linalg.norm(v1, axis=2) * np.linalg.norm(v2, axis=2))
    bond_ang = np.pad(np.arccos(np.clip(cosang, -1, 1)), ((0,0),(1,1)), mode="edge")
  
    # now look at time differences for velocity
    time_differences = np.diff(centroids, axis=0)
    features = np.concatenate([time_differences, centroids[1:], bond_len[1:,:, None], bond_ang[1:,:, None]], axis=2)

    timesteps = np.broadcast_to(np.arange(features.shape[0])[:, None], features.shape[:2])

    targets = residues * features.shape[0]

    features = features.reshape(-1, 8) # res * time, features
    timesteps = timesteps.flatten()
    
    features_df = pd.DataFrame(features, columns = ["x","y","z","dx","dy","dz","bond_len", "bond_ang"]) 
    metadata_df = pd.DataFrame(timesteps, columns=["timestep"], index = features_df.index) # this will be a groupby column
    targets_df = pd.DataFrame(targets, columns=["residue"], index = features_df.index)

    df = pd.concat([metadata_df, features_df, targets_df], axis=1)
    return df
    

def download_and_process_file(url, pdb_id):
    zip_file = os.path.join("md_data",f"{pdb_id}.zip")
    folder = os.path.join("md_data",f"{pdb_id}")
    os.makedirs(folder)
    try:
        with requests.get(url, stream=True) as r: # stream=True is very important
            r.raise_for_status()
            with open(zip_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)
        
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
        trajectories_df["pdb_id"] = pdb_id
        #trajectories_df.to_csv(os.path.join(folder, f"{pdb_id}.csv"))
        return trajectories_df
    except requests.exceptions.HTTPError as errh:
        print(f"failed to read {url}") 
        return pd.DataFrame()
    except requests.exceptions.ChunkedEncodingError as e:
        print(f"failed to read {url}") 
        return pd.DataFrame()
    





def main():
    base_url = "https://www.dsimb.inserm.fr/ATLAS/api" 
    atlas_df = pd.read_csv(os.path.abspath("atlas.csv")).iloc[:100]
    # for whatever reason the format for the api is slightly different
    atlas_df["pdb"] = atlas_df["pdb"].map(lambda x: x[:4] + "_" + x[-1])
    md_urls = [(base_url + f"/ATLAS/analysis/{pdb}", pdb) for pdb in atlas_df["pdb"].to_list()]

    out_dfs = []

    max_workers = 16
    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        futures = {
            executor.submit(download_and_process_file, url, pdb): pdb for url,pdb in md_urls
        }

        kwargs = {
            "total": len(futures),
            "desc": "Processing jobs",
            "unit": "job",
        }
        for future in tqdm(as_completed(futures), **kwargs):
            try:
                out_df = future.result()
                out_dfs.append(out_df)
            except Exception as e:
                print(f"Job generated an exception: {e}")

    out_df = pd.concat(out_dfs, axis=0, ignore_index=True)     
    out_df.to_csv(os.path.join("md_data","atlas_index.csv"))



if __name__ == "__main__":
    main()

"""



import shutil
import os
import requests
import mdtraj
import numpy as np
import pandas as pd
from zipfile import ZipFile
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def process_traj(traj):
    top = traj.topology
    n_frames = traj.n_frames
    n_res = top.n_residues
    
    res_indices = np.array([r.index for r in top.residues])
    
    heavy_atoms = top.select("not element H")
    heavy_res_ids = np.array([top.atom(idx).residue.index for idx in heavy_atoms])
    heavy_masses = np.array([top.atom(idx).element.mass for idx in heavy_atoms])
    
    centroids = np.empty((n_frames, n_res, 3), dtype=np.float32)
    
    for i, res_idx in enumerate(res_indices):
        mask = (heavy_res_ids == res_idx)
        if not np.any(mask):
            centroids[:, i, :] = 0.0
            continue
        sub_xyz = traj.xyz[:, heavy_atoms[mask], :]
        masses_sub = heavy_masses[mask]
        centroids[:, i, :] = np.sum(sub_xyz * masses_sub[None, :, None], axis=1) / masses_sub.sum()

    sequence_differences = np.diff(centroids, axis=1)
    bond_len = np.linalg.norm(sequence_differences, axis=2)
    bond_len = np.pad(bond_len, ((0,0),(0,1)), mode="edge")
    
    v1 = centroids[:, 1:-1] - centroids[:, :-2]
    v2 = centroids[:, 2:] - centroids[:, 1:-1]
    
    cosang = np.einsum('tij,tij->ti', v1, v2) / (np.linalg.norm(v1, axis=2) * np.linalg.norm(v2, axis=2) + 1e-8)
    bond_ang = np.pad(np.arccos(np.clip(cosang, -1.0, 1.0)), ((0,0),(1,1)), mode="edge")
    
    time_differences = np.diff(centroids, axis=0)
    
    features = np.concatenate([
        time_differences, 
        centroids[1:], 
        bond_len[1:, :, None], 
        bond_ang[1:, :, None]
    ], axis=2).reshape(-1, 8)
    
    timesteps = np.broadcast_to(np.arange(n_frames - 1)[:, None], (n_frames - 1, n_res)).flatten()
    res_targets = np.broadcast_to(res_indices[None, :], (n_frames - 1, n_res)).flatten()
    
    df = pd.DataFrame(features, columns=["x","y","z","dx","dy","dz","bond_len", "bond_ang"])
    df.insert(0, "timestep", timesteps)
    df["residue"] = res_targets
    
    return df

def download_and_process_file(url, pdb_id):
    base_md_dir = "md_data"
    os.makedirs(base_md_dir, exist_ok=True)
    
    # OPTIMIZATION: Work entirely inside an isolated scratch folder
    worker_temp_dir = os.path.join(base_md_dir, f"tmp_{pdb_id}")
    os.makedirs(worker_temp_dir, exist_ok=True)
    
    zip_file = os.path.join(worker_temp_dir, f"{pdb_id}.zip")
    extract_folder = os.path.join(worker_temp_dir, "extracted")
    
    try:
        # 1. Stream the download
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(zip_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
                    
        # 2. Extract files locally
        with ZipFile(zip_file, 'r') as zObject:
            zObject.extractall(path=extract_folder)
            
        traj_ids = [f"{pdb_id}_R{i}" for i in range(1, 4)]
        dfs = []
        
        # 3. Process into RAM
        for eye_d in traj_ids:
            xtc_path = os.path.join(extract_folder, f"{eye_d}.xtc")
            pdb_path = os.path.join(extract_folder, f"{pdb_id}.pdb")
            
            if not os.path.exists(xtc_path) or not os.path.exists(pdb_path):
                continue
                
            traj = mdtraj.load_xtc(xtc_path, top=pdb_path)
            df = process_traj(traj)
            df["traj_id"] = eye_d
            dfs.append(df)
            
        if not dfs:
            return pd.DataFrame()
            
        trajectories_df = pd.concat(dfs, axis=0, ignore_index=True)
        trajectories_df["pdb_id"] = pdb_id
        
        # Return the DataFrame directly to main memory
        return trajectories_df
        
    except Exception as e:
        print(f"Failed to process {pdb_id}: {str(e)}")
        return pd.DataFrame()
        
    finally:
        # OPTIMIZATION: Always wipe the entire temp folder immediately.
        # This keeps your hard drive perfectly clean while processing.
        if os.path.exists(worker_temp_dir):
            shutil.rmtree(worker_temp_dir, ignore_errors=True)

def main():
    base_url = "https://www.dsimb.inserm.fr/ATLAS/api"
    
    if not os.path.exists("atlas.csv"):
        print("Error: atlas.csv missing.")
        return
        
    atlas_df = pd.read_csv("atlas.csv")
    atlas_df["pdb"] = atlas_df["pdb"].map(lambda x: x[:4] + "_" + x[-1])
    md_urls = [(base_url + f"/ATLAS/analysis/{pdb}", pdb) for pdb in atlas_df["pdb"].to_list()]
    
    out_dfs = []
    max_workers = os.cpu_count()
    
    print(f"Starting pipeline using {max_workers} parallel workers...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_and_process_file, url, pdb): pdb for url, pdb in md_urls}
        
        kwargs = {"total": len(futures), "desc": "Processing PDB jobs", "unit": "job"}
        for future in tqdm(as_completed(futures), **kwargs):
            try:
                res_df = future.result()
                if not res_df.empty:
                    out_dfs.append(res_df)
            except Exception as e:
                print(f"Worker generated an exception: {e}")
                
    if out_dfs:
        print("Saving final concatenated dataset...")
        final_df = pd.concat(out_dfs, axis=0, ignore_index=True)
        final_df.to_csv(os.path.join("md_data", "atlas_index.csv"), index=False)
        print("Done!")
    else:
        print("No data processed successfully.")

if __name__ == "__main__":
    main()

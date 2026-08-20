import shutil
import os
import requests
import mdtraj
import numpy as np
import pandas as pd
from zipfile import ZipFile
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import traceback
from itertools import product

BACKBONE_ATOMS = ["CA", "N", "C", "O"]
FEATURE_COLUMNS= [atom_name+"_"+coord for atom_name,coord in product(BACKBONE_ATOMS + ["centroid"], ["x","y","z"])] + [ "phi","phi","omega"]

def process_traj(traj):
    top = traj.topology
    n_frames = traj.n_frames
    n_res = top.n_residues
    res_names = [res.name for res in top.residues]    
    res_indices = np.array([r.index for r in top.residues])
    all_atoms = [] 
    for i, res_idx in enumerate(res_indices):
        residue_atoms = []
        centroid = traj.xyz[:, top.select(f'resid {res_idx} and backbone')].mean(dim=1)
        for atom in BACKBONE_ATOMS:
            residue_atoms.append(traj.xyz[:, top.select(f'resid {res_idx} and name {atom} and backbone')].squeeze(1)) # n_frames,3
        all_atoms.append(np.concatenate(residue_atoms + [centroid], axis=1)) # n_frames, 12
    all_atoms = np.stack(all_atoms, axis=1) # n_frames, n_residues, 12 (there's a 1 dim at 2 for whatever reason)

    phi   = np.pad(mdtraj.compute_phi(traj)[1], ((0,0),(1,0)), mode="constant", constant_values=0.0)
    psi   = np.pad(mdtraj.compute_psi(traj)[1], ((0,0),(0,1)), mode="constant", constant_values=0.0)
    omega = np.pad(mdtraj.compute_omega(traj)[1], ((0,0),(0,1)), mode="constant", constant_values=0.0)
    angles_features = np.stack([phi,psi,omega], axis=2) # n_frames, n_residues-1, 3
    
    features = np.concatenate([all_atoms, angles_features], axis=2)
    timesteps = np.broadcast_to(np.arange(n_frames)[:,None], features.shape[:2])
    features = features.reshape(n_frames * n_res, -1, order="C") # C means right most columns will change the quickest

    timesteps = timesteps.flatten(order="C")
    res_targets = res_names * n_frames  
    df = pd.DataFrame({"residue":res_targets})
    df.insert(0, "timestep", timesteps)
    
    # limit the number of timesteps:
    timestep_mask = df["timestep"] < 200 
    df = df[timestep_mask]
    features = features[timestep_mask]

    return df, features 

def download_and_process_file(url, pdb_id):
    base_md_dir = "md_data"
    os.makedirs(base_md_dir, exist_ok=True)
    
    worker_temp_dir = os.path.join(base_md_dir, f"tmp_{pdb_id}")
    os.makedirs(worker_temp_dir, exist_ok=True)
    
    zip_file = os.path.join(worker_temp_dir, f"{pdb_id}.zip")
    extract_folder = os.path.join(worker_temp_dir, "extracted")
    
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(zip_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
                    
        with ZipFile(zip_file, 'r') as zObject:
            zObject.extractall(path=extract_folder)
            
        traj_ids = [f"{pdb_id}_R{i}" for i in range(1, 2)]# just look at first one, can load whole dataset this way
        data = []
        
        pdb_path = os.path.join(extract_folder, f"{pdb_id}.pdb")
        for eye_d in traj_ids:
            xtc_path = os.path.join(extract_folder, f"{eye_d}.xtc")
            
            if not os.path.exists(xtc_path) or not os.path.exists(pdb_path):
                continue
                
            traj = mdtraj.load_xtc(xtc_path, top=pdb_path)
            df, features = process_traj(traj)
            df["traj_id"] = eye_d
            data.append((df, features))
            
        if not data:
            return pd.DataFrame()
        dfs, features = zip(*data) 
        trajectories_df = pd.concat(dfs, axis=0, ignore_index=True)
        trajectories_df["pdb_id"] = pdb_id
        features_np = np.concatenate(features, axis=0)
        
        return trajectories_df, features_np
        
    except Exception as e:
        print(f"Failed to process {pdb_id}: {str(e)}")
        traceback.print_exception(e)
        return pd.DataFrame(), np.empty((0, len(FEATURE_COLUMNS)))

        
    finally:
        if os.path.exists(worker_temp_dir):
            shutil.rmtree(worker_temp_dir, ignore_errors=True)

def main(atlas_df, out_csv_name):
    base_url = "https://www.dsimb.inserm.fr/ATLAS/api"
       
    md_urls = [(base_url + f"/ATLAS/analysis/{pdb}", pdb) for pdb in atlas_df["pdb"].to_list()]
    
    out_data = []
    max_workers = os.cpu_count()
    
    print(f"Starting pipeline using {max_workers} parallel workers...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_and_process_file, url, pdb): pdb for url, pdb in md_urls}
        
        kwargs = {"total": len(futures), "desc": "Processing PDB jobs", "unit": "job"}
        for future in tqdm(as_completed(futures), **kwargs):
            try:
                res_df, out_np = future.result()
                if not res_df.empty:
                    out_data.append((res_df, out_np))
            except Exception as e:
                print(f"Worker generated an exception: {e}")
                
    if out_data:
        out_dfs, out_nps = zip(*out_data)
        
        final_df = pd.concat(out_dfs, axis=0, ignore_index=True)
        print(final_df.head())
        final_np = np.concatenate(out_nps, axis=0)
        final_df.to_csv(os.path.join("md_data", f"{out_csv_name}.csv"), index=False)
        np.save( os.path.join("md_data", f"{out_csv_name}.npy"), final_np)
    else:
        print("No data processed successfully.")

if __name__ == "__main__":
    atlas_df = pd.read_csv("atlas.csv")
    atlas_df["pdb"] = atlas_df["pdb"].map(lambda x: x[:4] + "_" + x[-1])
    df_len = len(atlas_df)

    main(atlas_df, f"atlas_index")

    #for i in range(16):
        #main(atlas_df.iloc[i * (df_len // 16) : (i+1)*(df_len // 16)], f"atlas_index_{i}")


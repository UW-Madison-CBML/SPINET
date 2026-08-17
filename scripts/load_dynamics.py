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
FEATURE_COLUMNS=["x","y","z","psi","phi","bond_len", "bond_ang"]
backbone_atoms = ["CA", "N", "C", "O"]

def process_traj(traj):
    top = traj.topology
    n_frames = traj.n_frames
    n_res = top.n_residues
    res_names = [res.name for res in top.residues]    
    res_indices = np.array([r.index for r in top.residues])
    all_atoms = [] 
    for i, res_idx in enumerate(res_indices):
        residue_atoms = []
        for atom in backbone_atoms:
            residue_atoms.append(traj.xyz[:, top.select(f'resid {res_idx} and name {atom} and backbone')])
        all_atoms.append(np.stack(residue_atoms, dim=0))
    all_atoms = np.stack(all_atoms, dim=0)
    _, phi   = md.compute_phi(traj)
    _, psi  = md.compute_psi(traj)
    _, omega = md.compute_omega(traj) 
    print(phi.shape)
    print(psi.shape)
    print(omega.shape)
            
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
        
        # Return the DataFrame directly to main memory
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
        print("Saving final concatenated dataset...")
        out_dfs, out_nps = zip(*out_data)
        
        final_df = pd.concat(out_dfs, axis=0, ignore_index=True)
        print(final_df.head())
        final_np = np.concat(out_nps, axis=0)
        final_df.to_csv(os.path.join("md_data", f"{out_csv_name}.csv"), index=False)
        np.save( os.path.join("md_data", f"{out_csv_name}.npy"), final_np)
        print("Done!")
    else:
        print("No data processed successfully.")

if __name__ == "__main__":
    atlas_df = pd.read_csv("atlas.csv")
    atlas_df["pdb"] = atlas_df["pdb"].map(lambda x: x[:4] + "_" + x[-1])
    df_len = len(atlas_df)

    main(atlas_df, f"atlas_index")

    #for i in range(16):
        #main(atlas_df.iloc[i * (df_len // 16) : (i+1)*(df_len // 16)], f"atlas_index_{i}")


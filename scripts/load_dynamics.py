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
import socket
import h5py

BACKBONE_ATOMS = ["CA", "N", "C", "O"] # this is the GT order of backbone atoms in a coordinates array
FRAME_ORIGIN = "CA"

def process_traj(traj,backbone_atoms, frame_origin): # save hdf5 and path to it
    atom_indices = {atom: i for i, atom in enumerate(backbone_atoms)}
    not_frame_origin_mask = np.array(backbone_atoms) != frame_origin

    top = traj.topology
    n_frames = traj.n_frames
    n_res = top.n_residues
    res_names = [res.name for res in top.residues]    
    res_indices = np.array([r.index for r in top.residues])
    all_atoms = [] 
    for i, res_idx in enumerate(res_indices):
        residue_atoms = []
        for atom in backbone_atoms:
            residue_atoms.append(traj.xyz[:, top.select(f'resid {res_idx} and name {atom} and backbone')].squeeze(1)) # n_frames, 3
        all_atoms.append(np.stack(residue_atoms, axis=1)) # n_frames, 4, 3
    all_atoms = np.stack(all_atoms, axis=0) # n_frames, n_residues, 4, 3
    phi_rad = mdtraj.compute_phi(traj)[1]
    psi_rad = mdtraj.compute_psi(traj)[1]
    omega_rad = mdtraj.compute_omega(traj)[1]

    phi_sin   = np.pad(np.sin(phi_rad), ((0,0),(1,0)), mode="constant", constant_values=0.0)
    psi_sin   = np.pad(np.sin(psi_rad), ((0,0),(0,1)), mode="constant", constant_values=0.0)
    omega_sin = np.pad(np.sin(omega_rad), ((0,0),(0,1)), mode="constant", constant_values=0.0)
    phi_cos   = np.pad(np.cos(phi_rad), ((0,0),(1,0)), mode="constant", constant_values=0.0)
    psi_cos   = np.pad(np.cos(psi_rad), ((0,0),(0,1)), mode="constant", constant_values=0.0)
    omega_cos = np.pad(np.cos(omega_rad), ((0,0),(0,1)), mode="constant", constant_values=0.0)

    angle_features = np.stack([phi_sin, phi_cos, psi_sin, psi_cos, omega_sin, omega_cos], axis=2).transpose(1,0,2) # n_frames, n_residues, 6

    carbon_alphas = all_atoms[:, :, atom_indices[frame_origin]]

    u,v = carbon_alphas - all_atoms[:, :, atom_indices["C"]], all_atoms[:, :, atom_indices["N"]] - carbon_alphas

    x_basis = u - v
    # normalize
    x_basis = x_basis / np.clip(np.linalg.norm(x_basis, axis = -1, keepdims=True), 0.01, None)

    y_basis = np.cross(u, v, axis = -1) 
    # normalize
    y_basis = y_basis / np.clip(np.linalg.norm(y_basis, axis=-1, keepdims=True), 0.01, None)
    
    # get final orthonormal basis vector:
    z_basis = np.cross(x_basis, y_basis, axis=-1) # will be normal since other two vectors are normal 
    
    basis_to_raw_matrix = np.stack([x_basis,y_basis,z_basis], axis=-1) # num_res, n_frames, 3, 3

    # features will be certain positions in the coordinate frame
    relative_features = all_atoms[:, :, not_frame_origin_mask, :] - all_atoms[:, :, ~not_frame_origin_mask, :] # num_res, n_frames, num_atoms-1, 3
    in_frame_features = np.matmul(relative_features, basis_to_raw_matrix).reshape(all_atoms.shape[0], all_atoms.shape[1], 3*(len(backbone_atoms)-1)) # num_res, n_frames, (num_atoms-1) * 3
    
    spinet_features = np.concatenate([in_frame_features, angle_features], axis=-1)
    
    return angle_features, all_atoms, spinet_features, basis_to_raw_matrix, np.array(res_names, "S4")

def download_and_process_file(url, pdb_id, cath_top, cath_hom):
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
        
        pdb_path = os.path.join(extract_folder, f"{pdb_id}.pdb")
        path = os.path.abspath(os.path.join(base_md_dir, f'{pdb_id}.h5'))
        with h5py.File(path,'w') as f:
            for eye_d in traj_ids:
                traj_group = f.create_group(f"{cath_top}/{cath_hom}/{pdb_id}/temp_1/{eye_d}")
                xtc_path = os.path.join(extract_folder, f"{eye_d}.xtc")
                
                if not os.path.exists(xtc_path) or not os.path.exists(pdb_path):
                    continue
                    
                traj = mdtraj.load_xtc(xtc_path, top=pdb_path)
                
                dihedrals, coordinates, spinet_features, frame_maps, residues = process_traj(traj, BACKBONE_ATOMS, FRAME_ORIGIN)

                traj_group.create_dataset("coordinates", data=coordinates)

                traj_group.create_dataset("dihedrals", data=dihedrals)

                traj_group.create_dataset("spinet_features", data=spinet_features)

                traj_group.create_dataset("frame_maps", data=frame_maps)

                traj_group.create_dataset("residues", data=residues)
           
        return path
        
    except Exception as e:
        print(f"Failed to process {pdb_id}: {str(e)}")
        return ""
        
    finally:
        if os.path.exists(worker_temp_dir):
            shutil.rmtree(worker_temp_dir, ignore_errors=True)

def main(atlas_df, out_csv_name):
    first_cath_lineage = [cath_id.strip().split("<br>")[0].split(".") for cath_id in atlas_df["cath_id"].to_list()] 

    _,_,top,hom = zip(*first_cath_lineage)

    atlas_df["cath_top"] = top
    atlas_df["cath_hom"] = hom

    print(atlas_df["cath_hom"].unique())

    base_url = "https://www.dsimb.inserm.fr/ATLAS/api"
       
    md_urls = [(base_url + f"/ATLAS/analysis/{row['pdb']}", row["pdb"], row["cath_top"], row["cath_hom"]) for _, row in atlas_df.iterrows()]
    
    out_data = []
    max_workers = os.cpu_count()
    
    print(f"Starting pipeline using {max_workers} parallel workers...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_and_process_file, url, pdb, cath_top, cath_hom): pdb for url, pdb, cath_top, cath_hom in md_urls}
        
        kwargs = {"total": len(futures), "desc": "Processing PDB jobs", "unit": "job"}
        for future in tqdm(as_completed(futures), **kwargs):
            try:
                data = future.result()
                out_data += data
            except Exception as e:
                print(f"Worker generated an exception: {e}")
    # now merge the files together
    

if __name__ == "__main__":
    # credit: Ian Stapleton Cordasco on StackOverflow
    socket.create_connection(('www.dsimb.inserm.fr', 443), timeout=3)

    atlas_df = pd.read_csv("atlas.csv")
    atlas_df["pdb"] = atlas_df["pdb"].map(lambda x: x[:4] + "_" + x[-1])
    df_len = len(atlas_df)

    main(atlas_df, f"atlas_index")

    #for i in range(16):
        #main(atlas_df.iloc[i * (df_len // 16) : (i+1)*(df_len // 16)], f"atlas_index_{i}")


import os
import multiprocessing as mp

import torch
from scrmsd import kabsch_rmsd
import h5py
from residue_classifier_dataset import ResidueClassifierDataset
import torch.nn.functional as F
import numpy as np
# we are only aligning backbone_atoms since that's all the model sees, i.e. num_atoms = |{N,CA,C,O}|

def get_traj_rmsd(traj, res_mask, i_idx, j_idx):
    """
    RMSD(i,j) == RMSD(j,i), so only compute the i<j pairs (upper triangle, diagonal is trivially 0)
    and mirror them into a full T,T matrix. Runs on a single trajectory so it can be handed to a
    worker process.

    traj: T, num_res, num_atoms, 3
    res_mask: num_res, bool (padding mask, same for every frame so it's 1D here)
    i_idx, j_idx: N, the upper-triangle (row, col) pairs to compute
    """
    torch.set_num_threads(1) # don't let each worker fight the others for cores

    T, num_res, num_atoms, n = traj.shape
    traj = traj.reshape(T, num_res * num_atoms, n)
    atom_mask = res_mask[:, None].expand(num_res, num_atoms).reshape(num_res * num_atoms)
    pair_mask = atom_mask[None, :].expand(i_idx.shape[0], -1)

    rmsd = kabsch_rmsd(traj[i_idx], traj[j_idx], pair_mask, device="cpu")

    rmsd_matrix = torch.zeros(T, T, dtype=rmsd.dtype)
    rmsd_matrix[i_idx, j_idx] = rmsd
    rmsd_matrix[j_idx, i_idx] = rmsd
    return rmsd_matrix.numpy()


def get_timewise_rmsd_pairs(trajs, mask):
    """
    trajs: B, T, num_res, num_atoms, 3
    mask: B, num_res, bool (padding mask, constant across time)
    """
    B, T = trajs.shape[0], trajs.shape[1]
    i_idx, j_idx = torch.triu_indices(T, T, offset=1) # only the strict upper triangle needs computing

    with mp.Pool(processes=min(os.cpu_count(), B)) as pool:
        rmsd_matrices = pool.starmap(
            get_traj_rmsd, [(trajs[b], mask[b], i_idx, j_idx) for b in range(B)]
        )

    return np.stack(rmsd_matrices)



if __name__ == "__main__":
    # i,j indexing upper triangle

    COURSE_GRAIN = 25 # since the sims are continuous, it's a waste to check immedatiately neighboring frames rather we can course grain it and use this index as to cluster
    # load data, i.e. get like the padded tensor trajs = T, num_res, 4, 3, where num_res is padded and mask reflects this
    trajs = []
    pdbs = []
    def visit(name, obj):
        if isinstance(obj, h5py.Group) and all(ds in obj for ds in ResidueClassifierDataset.REQUIRED_DATASETS):
            trajs.append(torch.from_numpy(obj["coordinates"][:, ::COURSE_GRAIN]))
            # pdbs = the list of pdbs that indexes dim_0 = B
            pdbs.append(name.split("/")[0]) # file system needs to be consistent here, at least at the start
    with h5py.File("atlas_data.h5", "r") as h5_file:
        h5_file.visititems(visit)
    res_nums = [traj.shape[0] for traj in trajs]
    max_res_num = max(res_nums) 
    trajs = torch.stack([F.pad(traj, (0,0, 0,0, 0,0, 0,max_res_num-length), "constant", 0) for traj, length in zip(trajs, res_nums)], dim=0)
    trajs = trajs.permute(0,2,1,3,4)
    mask = torch.arange(max_res_num)[None, :] < torch.tensor(res_nums)[:, None] # B, num_res; constant across time

    assert mask.shape == (trajs.shape[0], trajs.shape[2]), "mask residue dim does not match trajs"

    rmsd_np = get_timewise_rmsd_pairs(trajs, mask)
    # rmsd_np : shape should be B, T, T because of upper tri
    assert len(pdbs) == rmsd_np.shape[0]
    assert all(len(pdb) == 6 for pdb in pdbs), "pdbs are not constant size of 6 of format PPPP_C, pdbs will be silently chopped"
    pdbs = np.array(pdbs, dtype="S6") # pdb_id with chain should be fixed at 6
    with h5py.File("rmsd_cluster_index.h5", "w") as f:
        ds = f.create_dataset("cluster_index", data=rmsd_np)
        ds.attrs["pdbs"] = pdbs
        ds.attrs["course_grain"] = np.array(COURSE_GRAIN)
    print("saved cluster index")


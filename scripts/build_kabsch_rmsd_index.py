import torch
from scrmsd import kabsch_rmsd
import h5py
from residue_classifier_dataset import ResidueClassifierDataset
import torch.nn.functional as F
# we are only aligning backbone_atoms since that's all the model sees, i.e. num_atoms = |{N,CA,C,O}|
def get_timewise_rmsd_pairs(trajs, mask):
    """
    trajs: B, T, num_res, num_atoms, 3
    """
    B, T, num_res, num_atoms, n = trajs.shape
    # flatten the mask and trajs along the num_atoms dim. We want this to be one dim cause there is no structure that needs to be upheld other than atom m needs to be mapped to atom m in both molecules
    trajs = trajs.reshape(B,T, num_res * num_atoms, n) # we need the atom size dims to be 1 dim
    mask = mask[:,:,:,None].repeat(1,1,1,num_atoms).reshape(B, T, num_res * num_atoms)

    rows = trajs[:,None,:,:,:,:].expand(-1,T, -1, -1, -1, -1)
    cols = trajs[:,:,None,:,:,:].expand(-1,-1, T, -1, -1, -1)
    mask = mask[:,None,:,:].expand(-1,T, -1, -1)

    return kabsch_rmsd(rows, cols, mask, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")) # we can do this quickly on the GPU I think. It should only take up ~ 4 GB

if __name__ == "__main__":
    COURSE_GRAIN = 25 # since the sims are continuous, we should not check immedatiately neighboring frames rather we can course grain it and use this index as to cluster
    # load data, i.e. get like the padded tensor trajs = T, num_res, 4, 3, where num_res is padded and mask reflects this
    trajs = []
    pdbs = []
    with h5py.File("atlas_data.h5") as h5_file:
        def visit(name, obj):
            if isinstance(obj, h5py.Group) and all(ds in obj for ds in ResidueClassifierDataset.REQUIRED_DATASETS):
                trajs.append(torch.from_numpy(obj["coordinates"][:, ::COURSE_GRAIN]))
                # pdbs = the list of pdbs that indexes dim_0 = B
                pdbs.append(name.split("/")[0]) # file system needs to be consistent here, at least at the start
        h5_file.visititems(visit)
    res_nums = [traj.shape[0] for traj in trajs]
    max_res_num = max(res_nums) 
    trajs = torch.stack([F.pad(traj, (0,max_res_num-length, 0,0, 0,0, 0,0), "constant", 0) for traj, length in zip(trajs, res_nums)], dim=0)
    trajs = trajs.permute(0,2,1,3,4)
    mask = (torch.arange(max_res_num)[None, :] < torch.tensor(res_nums)[:, None])[:,None,:].repeat(1, trajs.shape[1], 1)

    assert mask.shape[:3] == trajs.shape[:3], "first 3 dims of mask do not match trajs"

    rmsd_tensor = get_timewise_rmsd_pairs(trajs, mask)
    
    rmsd_np = rmsd_tensor.numpy()
    # rmsd_np : shape should be B, T, T
    assert len(pdbs) == rmsd_np.shape[0]
    assert all(len(pdb) == 6 for pdb in pdbs), "pdbs are not constant size of 6 of format PPPP_C, pdbs will be silently chopped"
    pdbs = np.array(pdbs, dtype="S6") # pdb_id with chain should be fixed at 6
    with h5py.File("rmsd_cluster_index.h5", "w") as f:
        ds = f.create_dataset("cluster_index", data=rmsd_np)
        ds.attrs["pdbs"] = pdbs
        ds.attrs["course_grain"] = COURSE_GRAIN
    print("saved cluster index")


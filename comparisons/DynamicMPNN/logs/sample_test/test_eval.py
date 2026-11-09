import sys, os, random
from pathlib import Path
R = os.getcwd()
sys.path[:0] = [R, R + "/DynamicMPNN/src"]
import numpy as np, torch, h5py, hydra
import relaxed_pdb
from train import build_processed_dataset, run_validation, BACKBONE_ATOMS, DEVICE
from eval_ckpt import config_from_ckpt, DecoderNodeBatching, build_split_loader, load_pretrained_weights
from types import SimpleNamespace
torch.manual_seed(0); np.random.seed(0); random.seed(0)
S = Path(R)
ids = ["1ubq_A", "1crn_A"]
recs, fails = relaxed_pdb.load_relaxed_structures(ids, cache_dir=str(S/"pdbc"), verbose=False)
print("fails", fails)
h5p = S/"t.h5"
three = {v:k for k,v in relaxed_pdb.THREE_TO_ONE.items()}
with h5py.File(h5p, "w") as f:
    for pid, r in recs.items():
        c = r["coords"]  # R,4,3 (A)
        T = 60
        traj = c[:, None] + np.random.randn(c.shape[0], T, 4, 3).astype(np.float32) * 0.4; traj[:, 0] = c
        g = f.create_group(pid + "/traj")
        g["coordinates"] = traj
        g["residues"] = np.array([n.encode() for n in r["resnames"]])
        for d in ("dihedrals", "spinet_features", "frame_maps"): g[d] = np.zeros(1)
pdir = S/"proc"
build_processed_dataset(ids, h5p, pdir, pool_size=10, course_grain=4, force_rebuild=True)
ck = Path(R + "/DynamicMPNN/src").parents[0] / "checkpoints/single_chain_k2.ckpt"
mcfg, fcfg = config_from_ckpt(ck)
model = hydra.utils.instantiate(mcfg); load_pretrained_weights(model, ck)
model = DecoderNodeBatching(model).to(DEVICE).eval()
args = SimpleNamespace(pair_selection="tm_min", batch_size=2, num_workers=0)
loader, ds, order = build_split_loader(ids, fcfg, pdir, "test", args)
crit = torch.nn.CrossEntropyLoss()
# teacher-forced (old behaviour): call the wrapped model in train() mode path but no dropout
class TF(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, b):
        s.m.model.eval(); lg, vm = s.m.model(b)
        c = b.num_decoder_nodes.reshape(-1); b.batch = torch.repeat_interleave(torch.arange(c.numel(), device=lg.device), c.to(lg.device))
        return lg, vm
old = run_validation(TF(model), loader, crit, order, 0, None, split_name="test")
new = run_validation(model, loader, crit, order, 0, None, split_name="test")
for k in ("test_top1_acc_mean", "test_perp_mean"):
    print(k, "teacher-forced", round(float(old[k]), 4), "sampled", round(float(new[k]), 4))
t = new["test_pred_seqs"].get_dataframe() if hasattr(new["test_pred_seqs"], "get_dataframe") else None
for pid in order:
    print(pid, "native len", len(recs[pid]["seq"]))
if t is not None:
    for _, row in t.iterrows(): print(row["pdb"], "design len", len(row["seq"]), row["seq"][:40])

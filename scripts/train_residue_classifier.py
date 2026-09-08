import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch_geometric.data import Data

from residue_classifier_dataset import ResidueClassifierDataset
from invariant_features_sheaf_model import NodeSheafClassifier
import itertools
from itertools import product
from contextlib import nullcontext
from load_dynamics import BACKBONE_ATOMS
from Bio.SeqUtils import seq1

from colabfold.batch import get_queries
from alphafold.common import residue_constants

FEATURE_COLUMNS= [atom_name+"_"+coord for atom_name,coord in product(BACKBONE_ATOMS,["x","y","z"])] + [ "phi","psi","omega"]
import time

import json
import subprocess

from scrmsd import evaluate_batch_rmsd, ColabFoldValidationEngine


from huggingface_hub import login, HfApi, hf_hub_url, hf_hub_download, create_repo
from stats_utils import get_confusion_matrix, top_k_acc
from sheaf_utils import sheaf_laplacian
from torch.profiler import profile, ProfilerActivity, record_function
import h5py

def load_df_from_pdbs(local_path, file_name_format="p_c-t"):
    files = [path for path in os.listdir() if path.endswith(".pdb")] 
    
def run_val(run, model, loader, dataset, epoch, device, crit, colabfold_model, val_name="val", num_classes = len(ResidueClassifierDataset.AMINO_ACIDS), cm_title="Amino Acid Confusion Matrix", test_val=False):
    model.eval()

    global_confusion_mat = torch.zeros((num_classes, num_classes))
    acc_top_1 = []
    acc_top_5 = []
    acc_top_10 = []

    scrmsd = []

    losses = []
    f1s = {acid:[] for acid in ResidueClassifierDataset.AMINO_ACIDS}
    precisions = {acid:[] for acid in ResidueClassifierDataset.AMINO_ACIDS}
    recalls = {acid:[] for acid in ResidueClassifierDataset.AMINO_ACIDS}
    
    with torch.no_grad():
        for batch in tqdm(loader if not test_val else itertools.islice(loader,100), desc=f"Epoch {epoch} {val_name}", leave=False):

            batch = batch.to(device)
            batch = batch.sort()
            out_batch = model(batch)                                    # out_batch.x: (N, num_classes) logits, N = total nodes (residues) across all B proteins in the batched graph

            pred_mask = ~batch.node_mask.bool()                         # (N,) bool, True at nodes whose residue identity is masked (i.e. a prediction target)

            targets = batch.y[pred_mask]                                # (N_masked,) ground-truth class index, masked nodes only
            logits = out_batch.x[pred_mask]                             # (N_masked, num_classes)

            out_batch = out_batch.cpu()
            batch = batch.cpu()
            data_list = out_batch.to_data_list()                        # list of B per-protein Data objects (model output)
            gt_list = batch.to_data_list()                              # list of B per-protein Data objects (ground truth)

            pred_seqs = {}

            for i, gt_data in enumerate(gt_list):
                pred_data = data_list[i]

                pred_idx = pred_data.x.argmax(dim=-1)                   # (R_i,) predicted class index per residue, protein i (R_i = residue count, varies per protein)

                graph_mask = ~gt_data.node_mask.bool()                  # (R_i,) True at masked (prediction-target) residues

                pred_idx[~graph_mask] = gt_data.y[~graph_mask]          # unmasked residues get their known ground-truth identity instead of the prediction

                seq_str = "".join([seq1(ResidueClassifierDataset.AMINO_ACIDS[idx.item()]) for idx in pred_idx])  # length-R_i amino-acid string
                pred_seqs[gt_data.traj_id] = seq_str                    # accumulates to B entries, insertion order == gt_list order

            # Get gt coordinates from trajectory
            trajs = [[dataset.groups[prot.index[0][0]][1].get_group(idx) for idx in range(prot.index[0][1], prot.index[0][2])] for prot in gt_list]  # B lists of T per-frame DataFrames, each frame has R_i residue rows
            traj_tensors = []
            for traj in trajs:
                traj_atoms = []
                for atom in BACKBONE_ATOMS:
                    cols = [atom + "_" + coord for coord in ["x","y","z"]]
                    traj_atoms.append(torch.stack([torch.from_numpy(conf[cols].to_numpy()) for conf in traj ], dim=0)) # T, R, 3

                traj_tensor = torch.stack(traj_atoms, dim=2) # T, R, A, 3
                traj_tensors.append(traj_tensor)                        # B tensors, each (T, R_i, A, 3); R_i still ragged across proteins at this point
            lengths = torch.tensor([traj_tensor.shape[1] for traj_tensor in traj_tensors])  # (B,) residue count R_i per protein
            pad_size = max(lengths)                                     # scalar = max R_i in this batch


            traj_tensors = [F.pad(traj_tensor, (0,0,0,0,0,pad_size-traj_tensor.shape[1],0,0), mode="constant", value=0.0) for traj_tensor in traj_tensors]

            backbone_tensor = torch.stack(traj_tensors, dim=0)          # (B, T, pad_size, A, 3)
            gt_seq_mask = lengths[:,None] > torch.arange(pad_size)[None,:]  # (B, pad_size) bool

            out_batch = out_batch.cpu()
            batch = batch.cpu()
            preds = logits.argmax(dim=-1).cpu()                        # (N_masked,)
            targets_cpu = targets.cpu()                                 # (N_masked,)


            for pred_prot, gt_prot in zip(data_list, gt_list):

                pred_mask = ~gt_prot.node_mask                          # (R_i,) per-protein version of the batch-level pred_mask above

                log = pred_prot.x[pred_mask]                             # (R_i_masked, num_classes)
                targ = gt_prot.y[pred_mask]                              # (R_i_masked,)
                loss = crit(log, targ)                                  # scalar
                losses.append(loss.item())

                acc_top_1.append(top_k_acc(log, targ, 1))
                acc_top_5.append(top_k_acc(log, targ, 5))
                acc_top_10.append(top_k_acc(log, targ,10))

            batch_conf_mat = get_confusion_matrix(targets_cpu, preds, num_classes)  # (num_classes, num_classes)
            global_confusion_mat += batch_conf_mat

            diag = batch_conf_mat.diag()                                          # (num_classes,)
            recall = torch.nan_to_num(diag / batch_conf_mat.sum(dim=1), 0.0)      # (num_classes,)
            precision = torch.nan_to_num(diag / batch_conf_mat.sum(dim=0), 0.0)   # (num_classes,)
            f1 = torch.nan_to_num(2 * (precision * recall) / (precision + recall), 0.0)  # (num_classes,)


            for k, amino_acid in enumerate(ResidueClassifierDataset.AMINO_ACIDS):
                 precisions[amino_acid].append(precision[k].item())
                 recalls[amino_acid].append(recall[k].item())
                 f1s[amino_acid].append(f1[k].item())

            # Calculate rmsd per-batch (evaluate_batch_rmsd scores one structure per protein,
            # so collapse the trajectory-frame axis down to frame 0)
            gt_frame0_coords = backbone_tensor[:, 0]  # (B, T, pad_size, A, 3) -> (B, pad_size, A, 3)
            scrmsd.append(evaluate_batch_rmsd(list(pred_seqs.values()), gt_frame0_coords, gt_seq_mask, colabfold_model))



    prf_dict = {}
    precisions = {key: torch.tensor(value) for key, value in precisions.items()}  # each value: (num_batches,), one entry per batch this amino acid appeared in
    recalls = {key: torch.tensor(value) for key, value in recalls.items()}        # each value: (num_batches,)
    f1s = {key: torch.tensor(value) for key, value in f1s.items()}               # each value: (num_batches,)
    for k, amino_acid in enumerate(ResidueClassifierDataset.AMINO_ACIDS):
        prf_dict[f"{val_name}_{amino_acid}_f1_mean"] = f1s[amino_acid].mean().item()
        prf_dict[f"{val_name}_{amino_acid}_precision_mean"] = precisions[amino_acid].mean().item()
        prf_dict[f"{val_name}_{amino_acid}_recall_mean"] = recalls[amino_acid].mean().item()
        prf_dict[f"{val_name}_{amino_acid}_f1_std"] = f1s[amino_acid].std().item()
        prf_dict[f"{val_name}_{amino_acid}_precision_std"] = precisions[amino_acid].std().item()
        prf_dict[f"{val_name}_{amino_acid}_recall_std"] = recalls[amino_acid].std().item()

    # perplexity score
    perplexities = torch.exp(torch.tensor(losses))  # losses/perplexities: (total_proteins,) -- one scalar per protein, accumulated across every batch in the loader
    prf_dict[f"{val_name}_perp_mean"] = (pm := perplexities.mean().item())
    prf_dict[f"{val_name}_perp_std"] = (ps := perplexities.std().item())

    scrmsd_keys = ["C_rmsd", "CA_rmsd", "N_rmsd", "O_rmsd", "all_backbone_rmsd"]
    scrmsd = {key: torch.cat([batch_scores[key] for batch_scores in scrmsd]) for key in scrmsd_keys}  # each value: (total_proteins,)
    for key in scrmsd_keys:
        prf_dict[f"{val_name}_{key}_mean"] = scrmsd[key].mean().item()
        prf_dict[f"{val_name}_{key}_std"] = scrmsd[key].std().item()
    rmsdM = scrmsd["all_backbone_rmsd"].mean().item()
    rmsdS = scrmsd["all_backbone_rmsd"].std().item()

    acc_top_1 = torch.tensor(acc_top_1)
    acc_top_5 = torch.tensor(acc_top_5)
    acc_top_10 = torch.tensor(acc_top_10)

    prf_dict[f"{val_name}_top1_acc_mean"] = (a1m := acc_top_1.mean().item())
    prf_dict[f"{val_name}_top5_acc_mean"] = (a5m := acc_top_5.mean().item())
    prf_dict[f"{val_name}_top10_acc_mean"] = (a10m := acc_top_10.mean().item())
    prf_dict[f"{val_name}_top1_acc_std"] = (a1s := acc_top_1.std().item())
    prf_dict[f"{val_name}_top5_acc_std"] = (a5s := acc_top_5.std().item())
    prf_dict[f"{val_name}_top10_acc_std"] = (a10s := acc_top_10.std().item())

    print(f"{sum(p.numel() for p in model.parameters() if p.requires_grad)} & ${a1m:.3f} \\pm {a1s:.3f}$ & ${a5m:.3f} \\pm {a5s:.3f}$ & ${a10m:.3f} \\pm {a10s:.3f}$ & ${pm:.3f} \\pm {ps:.3f}$ & ${rmsdM: .3f} \\pm {rmsdS: .3f}$") 

    fig, ax = plt.subplots(figsize=(12, 12))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=global_confusion_mat.numpy().astype(int),
        display_labels=ResidueClassifierDataset.AMINO_ACIDS
    )
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.title(cm_title)

    prf_dict[f"{val_name}_aa_confusion_matrix"] = wandb.Image(fig)
    plt.close(fig)

    avg_loss = sum(losses) / len(losses) if losses else 0

    run.log(prf_dict | {f"epoch_{val_name}_loss": avg_loss, "epoch": epoch})

def interpret_sheaves(loader, model, run, device):
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Loading Sheaves", leave=False):

            data = batch.to_data_list()[0].to(device)
            data = data.sort() 
            _, first_sheaf, last_sheaf = model(data, return_sheaf=True)
            first_sheaf = first_sheaf.cpu()
            last_sheaf = last_sheaf.cpu()
            edge_index = data.edge_index.cpu()
            traj_id = data.traj_id if isinstance(data.traj_id, str) else data.traj_id[0]
            first_sheaf_laplacian = sheaf_laplacian(data.x.shape[0], first_sheaf, edge_index)
            last_sheaf_laplacian = sheaf_laplacian(data.x.shape[0], last_sheaf, edge_index)
            first_eigs = torch.linalg.eig(first_sheaf_laplacian)
            last_eigs = torch.linalg.eig(last_sheaf_laplacian)
            # we just need the real components here, sheaf laplacian is real positive semi-definite
            first_eigvals = first_eigs.eigenvalues.real
            first_eigvecs = first_eigs.eigenvectors.real 
            last_eigvals = first_eigs.eigenvalues.real
            last_eigvecs = first_eigs.eigenvectors.real

            img_dict = {}
            fig, ax = plt.subplots()  
            ax.plot(np.arange(first_eigvals.shape[0]),first_eigvals.numpy())
            img_dict["first_sheaf"] = wandb.Image(fig)
            plt.close(fig)

            fig, ax = plt.subplots()  
            ax.plot(np.arange(last_eigvals.shape[0]), last_eigvals.numpy())
            img_dict["last_sheaf"] = wandb.Image(fig)
            plt.close(fig)
            
            run.log(img_dict)
            
            
         

# TODO fix the train val test split
# move validation code to it's own function
def train_residue_classifier(args_dict):
    # hyperparameters
    epsilon = 5.0 # in Angstroms
    learning_rate = 1e-3
    epochs = args_dict['epochs']
    val_ratio = 0.15
    test_ratio = 0.15
    batch_size = 32
    hidden_dim = 64
    stalk_dim = 16
    num_blocks = 8
    masking_ratio = 1.0
    use_masking = masking_ratio < 1.0
    ablate_sheaves=args_dict["ablate_sheaves"]
    use_attention = not args_dict["ablate_attention"]
    num_heads = 4 if use_attention else 1
    run_name = args_dict["run_name"]
    restriction_map_type=args_dict["restriction_map_type"]
    seed=42
    paradigm = args_dict["paradigm"]
    num_timesteps = 128 if paradigm == "dynamic" else 1
    use_scheduler=False
    test_val = False
    use_profiler = False
    resume_model_name = args_dict["resume"]
    resume = args_dict["resume"] != ""
    nth_cross_val = args_dict["cross_val"]
    assert nth_cross_val >= 0 and nth_cross_val < 5
    
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 

    # login on HF
    login(token=os.environ["HF_TOKEN"])
    api = HfApi()

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch_rng = torch.Generator(); torch_rng = torch_rng.manual_seed(seed)
    np_rng = np.random.default_rng(seed=seed)

    # set up device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    h5_path = os.path.abspath("atlas_data.h5")
    with h5py.File(h5_path, "r") as f:
        all_pdbs = list(f.keys())
        np_rng.shuffle(all_pdbs)
    num_pdbs = len(all_pdbs) 
    group_size = num_pdbs // 5
    val_pdbs = all_pdbs[group_size * nth_cross_val: group_size * (nth_cross_val + 1)]
    train_pdbs = list(set(all_pdbs).difference(set(val_pdbs)))
    print("train: ", train_pdbs)
    print("val: ", val_pdbs)

    # set up wandb
    wandb.login(key=os.getenv("WANDB_KEY"))
    run = wandb.init(
        entity="jenslundsgaard7-uw-madison",
        project="SheafProtein",
        name=run_name,
        config={
            "epsilon": epsilon,
            "lr": learning_rate,
            "epochs": epochs,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "batch_size": batch_size,
            "hidden_dim": hidden_dim,
            "masking_ratio": masking_ratio,
            "task":"predicting residues from motions",
            "ablate_sheaves":ablate_sheaves,
            "stalk_dim": stalk_dim,
            "seed":seed,
            "trajectory_subsequence_timesteps":num_timesteps,
            "use_scheduler":use_scheduler,
            "scheduler_type":"cosine annealing warm restarts every epoch" if use_scheduler else "none",
            "restriction_map_type":restriction_map_type,
            "paradigm":paradigm,
            "use_masking": use_masking
        },
    )

    # WANDB artifact logging
    artifact = wandb.Artifact(name="scripts", type="model_file")
    artifact.add_file(os.path.abspath(__file__))

    dependencies = [ # TODO fix
        "invariant_features_sheaf_model.py",
        "residue_classifier_dataset.py",
        "sheaf_utils.py"
    ]

    for file in dependencies:
        if os.path.exists(file):
            artifact.add_file(os.path.abspath(file))

    run.log_artifact(artifact)

    # Initialize datasets
    train_dataset = ResidueClassifierDataset(h5_path, np_rng, groups=train_pdbs,  epsilon=epsilon, paradigm=paradigm, fixed_length=num_timesteps, traj_len=1000)
    val_dataset = ResidueClassifierDataset(h5_path, np_rng, groups=val_pdbs, epsilon=epsilon, paradigm=paradigm, fixed_length=num_timesteps, traj_len=1000)

    # set up dataloaders
    train_loader = DataLoader(train_dataset, shuffle=True, generator=torch_rng, batch_size=batch_size, num_workers=16, persistent_workers=True, worker_init_fn=ResidueClassifierDataset.worker_init_fn, collate_fn=lambda batch:train_dataset.graph_collate(batch), pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=16, collate_fn=lambda batch:val_dataset.graph_collate(batch), pin_memory=True, drop_last=False)

    single_graph_val_loader = itertools.islice(DataLoader(val_dataset, shuffle=True, generator=torch_rng, batch_size=1, num_workers=16, collate_fn=lambda batch:val_dataset.graph_collate(batch), pin_memory=True, drop_last=False), 100)

    num_classes = len(ResidueClassifierDataset.AMINO_ACIDS)

    # set up new diffusion model # TODO fix all this
    # ---------------------------------------------

    model = NodeSheafClassifier(
        paradigm=paradigm,
        atoms=BACKBONE_ATOMS,
        frame_origin="CA",
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        stalk_dim=stalk_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        ablate_sheaves=ablate_sheaves,
        num_timesteps=num_timesteps,
        restriction_map_type=restriction_map_type,
        use_attention=use_attention,
        use_masking=use_masking
    ).to(DEVICE)
    # Credit: Tomerikoo and Fabio Perez on StackOverflow
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run.log({"params":pytorch_total_params})

    create_repo(f"JensLundsgaard/{run_name}", exist_ok=True)
    if resume:
        weights_path = hf_hub_download("JensLundsgaard/" + resume_model_name, "pytorch_model.bin", local_dir=os.path.abspath("./"))
        model.load_state_dict(torch.load(weights_path, weights_only=True))

    # -----------------------------------------

    local_dir = f"./{run_name}"
    os.makedirs(local_dir, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    crit = torch.nn.CrossEntropyLoss() 
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, len(train_loader))

    # Instatiate colabfold model
    colabfold_model = ColabFoldValidationEngine(BACKBONE_ATOMS, device=DEVICE)

    # training loop
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader if not test_val else itertools.islice(train_loader,100), desc=f"Epoch {epoch+1}/{epochs} [Train]")
        times1 = []
        times2 = []
        times3 = []
        times4 = []
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities += [ProfilerActivity.CUDA]
        with (profile(activities=activities, record_shapes=True) if use_profiler else nullcontext()) as prof:
            for batch in pbar:
                if use_profiler:
                    torch.cuda.synchronize()
                times1.append(time.perf_counter())
                batch = batch.to(DEVICE)

                if use_profiler:
                    torch.cuda.synchronize()

                times2.append(time.perf_counter())
                optimizer.zero_grad()
                
                out_batch = model(batch)

                if use_profiler:
                    torch.cuda.synchronize()
                times3.append(time.perf_counter())

                pred_mask = ~batch.node_mask.bool()

                loss = crit(out_batch.x[pred_mask], batch.y[pred_mask])
                run.log({"train_loss": loss.item(), "epoch": epoch})

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if use_profiler:
                    torch.cuda.synchronize()

                times4.append(time.perf_counter())
                if use_scheduler:
                    scheduler.step()
                if use_profiler:
                    prof.step()
        
        if use_profiler:
            table = prof.key_averages()
            prof_results = prof.key_averages()
            df = pd.DataFrame(map(vars, prof_results))
            run.log({"profiler": wandb.Table(dataframe=df)})
        times1 = torch.tensor(times2) - torch.tensor(times1)
        times2 = torch.tensor(times3) - torch.tensor(times2)
        times3 = torch.tensor(times4) - torch.tensor(times3)
        batch_cycle_times = times1.diff()
        run.log({"to_gpu_time":times1.mean().item(), "forward_pass_time":times2.mean().item(), "backpass_time":times3.mean().item(), "cycle_time":batch_cycle_times.mean().item()}) 
        # push to hub

        #model.save_pretrained(local_dir, safe_serialization=False) # TODO fix this is bugged
        torch.save(model.state_dict(), os.path.join(local_dir, "pytorch_model.bin"))

        if hasattr(model, "config") and model.config is not None:
            config_path = os.path.join(local_dir, "config.json")
            with open(config_path, "w") as f:
                if hasattr(model.config, "to_dict"):
                    json.dump(model.config.to_dict(), f, indent=2)
                else:
                    json.dump(model.config, f, indent=2)
        api.upload_folder(
            folder_path=local_dir,
            repo_id=f"JensLundsgaard/{run_name}",
            repo_type="model"
        )
        
        # Validation Check
        run_val(run, model, val_loader, val_dataset, epoch, DEVICE, crit, colabfold_model, val_name="val", test_val=test_val)

        if not ablate_sheaves:
            interpret_sheaves(single_graph_val_loader, model, run, DEVICE)

    run.finish()

if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser(
        prog='Train sheaf protein node residue classifier',
        description='Trains sheaf node classifier to predict nodes') 
    parser.add_argument('--run-name', type=str, default="residue_classifier")
    parser.add_argument('--ablate-sheaves', action="store_true")
    parser.add_argument('--ablate-attention', action="store_true")
    parser.add_argument('--restriction-map-type', type=str, default="low_rank", choices=['low_rank', 'orthogonal', 'arbitrary'])
    parser.add_argument('--paradigm', type=str, default="dynamic", choices = ['dynamic', 'static', 'ensemble'])
    parser.add_argument('--resume', type=str, default="")
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--cross-val', type=int, default=0)
    args = parser.parse_args()
    train_residue_classifier(vars(args))

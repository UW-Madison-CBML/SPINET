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

from load_dynamics import BACKBONE_ATOMS
from Bio.SeqUtils import seq1

from colabfold.batch import get_queries
from alphafold.common import residue_constants

FEATURE_COLUMNS= [atom_name+"_"+coord for atom_name,coord in product(BACKBONE_ATOMS,["x","y","z"])] + [ "phi","psi","omega"]
import time

import json
import subprocess

from scrmsd import evaluate_batch_rmsd, ColabFoldValidationEngine
from huggingface_hub import login, HfApi
from stats_utils import get_confusion_matrix, top_k_acc


def train_residue_classifier(args_dict):
    # hyperparameters
    epsilon = 5.0 # in Angstroms
    learning_rate = 1e-3
    epochs = 8
    val_ratio = 0.15
    test_ratio = 0.15
    batch_size = 16
    hidden_dim = 16
    stalk_dim = 8
    num_blocks = 4
    masking_ratio = 0.75
    ablate_sheaves=args_dict["ablate_sheaves"]
    use_attention = not args_dict["ablate_attention"]
    num_heads = 4 if use_attention else 1
    run_name = args_dict["run_name"]
    restriction_map_type=args_dict["restriction_map_type"]
    seed=42
    num_timesteps = 128
    use_scheduler=False
    test_val = False
    
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 
    # login on HF
    login(token=os.environ["HF_TOKEN"])
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch_rng = torch.Generator(); torch_rng = torch_rng.manual_seed(seed)
    np_rng = np.random.default_rng(seed=seed)

    # set up device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load in data and create 3-way split
    df = pd.read_csv(os.path.join("md_data","atlas_index.csv"))

    features_np = np.load(os.path.join("md_data","atlas_index.npy"))

    # mask
    df["mask"] = np_rng.random(len(df)) > masking_ratio

    pdb_ids = df["pdb_id"].unique()
    num_pdbs = len(pdb_ids)

    val_cutoff = int(val_ratio * num_pdbs)
    test_cutoff = val_cutoff + int(test_ratio * num_pdbs)

    val_pdbs = pdb_ids[:val_cutoff]
    test_pdbs = pdb_ids[val_cutoff:test_cutoff]
    train_pdbs = pdb_ids[test_cutoff:]

    train_mask = df["pdb_id"].isin(train_pdbs)
    train_np = features_np[train_mask]
    train_df = df[train_mask]
    train_df = pd.concat([train_df, pd.DataFrame(train_np, columns=FEATURE_COLUMNS, index=train_df.index)], axis=1)

    val_mask = df["pdb_id"].isin(val_pdbs)
    val_np = features_np[val_mask]
    val_df = df[val_mask]
    val_df = pd.concat([val_df, pd.DataFrame(val_np, columns=FEATURE_COLUMNS, index=val_df.index)], axis=1)

    test_mask = df["pdb_id"].isin(test_pdbs)
    test_np = features_np[test_mask]
    test_df = df[test_mask]
    test_df = pd.concat([test_df, pd.DataFrame(test_np, columns=FEATURE_COLUMNS, index=test_df.index)], axis=1)

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
            "restriction_map_type":restriction_map_type
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
    train_dataset = ResidueClassifierDataset(train_df,  epsilon=epsilon, fixed_length=num_timesteps, traj_len=200)
    val_dataset = ResidueClassifierDataset(val_df, epsilon=epsilon, fixed_length=num_timesteps, traj_len=200)
    test_dataset = ResidueClassifierDataset(test_df, epsilon=epsilon, fixed_length=num_timesteps, traj_len=200)

    # set up dataloaders
    train_loader = DataLoader(train_dataset, shuffle=True, generator=torch_rng, batch_size=batch_size, num_workers=16, persistent_workers=True, worker_init_fn=ResidueClassifierDataset.worker_init_fn, collate_fn=lambda batch:train_dataset.graph_collate(batch), pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=16, collate_fn=lambda batch:val_dataset.graph_collate(batch), pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, num_workers=16, collate_fn=lambda batch:test_dataset.graph_collate(batch), pin_memory=True, drop_last=False)

    single_graph_val_loader = itertools.islice(DataLoader(val_dataset, shuffle=True, generator=torch_rng, batch_size=1, num_workers=16, collate_fn=lambda batch:val_dataset.graph_collate(batch), pin_memory=True, drop_last=False), 100)

    num_classes = len(ResidueClassifierDataset.AMINO_ACIDS)

    # set up new diffusion model # TODO fix all this
    # ---------------------------------------------

    model = NodeSheafClassifier(
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        stalk_dim=stalk_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        ablate_sheaves=ablate_sheaves,
        num_timesteps=num_timesteps,
        restriction_map_type=restriction_map_type,
        use_attention=use_attention
    ).to(DEVICE)

    # -----------------------------------------

    local_dir = f"./{run_name}"
    os.makedirs(local_dir, exist_ok=True)

    api = HfApi()


    colabfold_model = ColabFoldValidationEngine(BACKBONE_ATOMS, device=DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    crit = torch.nn.CrossEntropyLoss() 
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, len(train_loader))
    # profiler stuff
    table_path = os.path.abspath("table.txt")
    trace_path = os.path.abspath("trace.json")


    # training loop
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader if not test_val else itertools.islice(train_loader,100), desc=f"Epoch {epoch+1}/{epochs} [Train]")
        times1 = []
        times2 = []
        times3 = []
        times4 = []
        for batch in pbar:
            times1.append(time.perf_counter())
            batch = batch.to(DEVICE)

            batch = batch.sort()
            times2.append(time.perf_counter())
            optimizer.zero_grad()
            
            out_batch = model(batch)

            times3.append(time.perf_counter())

            pred_mask = ~batch.node_mask.bool()

            loss = crit(out_batch.x[pred_mask], batch.y[pred_mask])
            run.log({"train_loss": loss.item(), "epoch": epoch})

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            times4.append(time.perf_counter())
            if use_scheduler:
                scheduler.step()

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
        model.eval()

        global_confusion_mat = torch.zeros((num_classes, num_classes))
        val_acc_top_1 = []
        val_acc_top_5 = []
        val_acc_top_10 = []

        pos_correct = torch.zeros(0, dtype=torch.long)
        pos_total = torch.zeros(0, dtype=torch.long)

        val_losses = []
        f1s = {acid:[] for acid in ResidueClassifierDataset.AMINO_ACIDS}
        precisions = {acid:[] for acid in ResidueClassifierDataset.AMINO_ACIDS}
        recalls = {acid:[] for acid in ResidueClassifierDataset.AMINO_ACIDS}
        
        #scRMSDs = [] 
        with torch.no_grad():
            for batch in tqdm(val_loader if not test_val else itertools.islice(val_loader,100), desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):

                batch = batch.to(DEVICE)
                batch = batch.sort() 
                out_batch = model(batch)

                pred_mask = ~batch.node_mask.bool()

                targets = batch.y[pred_mask]
                logits = out_batch.x[pred_mask]

                """out_batch = out_batch.cpu()
                batch = batch.cpu()
                data_list = out_batch.to_data_list()
                gt_list = batch.to_data_list()

                pred_seqs = {}
                
                for i, gt_data in enumerate(gt_list):
                    pred_data = data_list[i]
    
                    pred_idx = pred_data.x.argmax(dim=-1)
                    
                    graph_mask = ~gt_data.node_mask.bool()
                    
                    pred_idx[~graph_mask] = gt_data.y[~graph_mask]
                    
                    seq_str = "".join([seq1(ResidueClassifierDataset.AMINO_ACIDS[idx.item()]) for idx in pred_idx])
                    pred_seqs[gt_data.traj_id] = seq_str
                trajs = [[val_dataset.groups[prot.index[0][0]][1].get_group(idx) for idx in range(prot.index[0][1], prot.index[0][2])] for prot in gt_list]
                traj_tensors = []
                for traj in trajs:
                    traj_atoms = []
                    for atom in BACKBONE_ATOMS:
                        cols = [atom + "_" + coord for coord in ["x","y","z"]]
                        traj_atoms.append(torch.stack([torch.from_numpy(conf[cols].to_numpy()) for conf in traj ], dim=0)) # T, R, 3
                        
                    traj_tensor = torch.stack(traj_atoms, dim=2) # T, R, A, 3
                    traj_tensors.append(traj_tensor)
                lengths = torch.tensor([traj_tensor.shape[1] for traj_tensor in traj_tensors])
                pad_size = max(lengths)
                traj_tensors = [F.pad(traj_tensor, (0,0,0,pad_size-traj_tensor.shape[1],0,0,0,0), mode="constant", value=0.0) for traj_tensor in traj_tensors]
                backbone_tensor = torch.stack(traj_tensors, dim=0)
                mask = lengths[:,None] < torch.arange(pad_size)[None,:]
                #scRMSD_results = evaluate_batch_rmsd(pred_seqs.values(), backbone_tensor, mask, colabfold_model)
                
                #scRMSDs.append(scRMSD_results["all_backbone_rmsd"].cpu().mean().item()) 
                
                """ 
                out_batch = out_batch.cpu()
                batch = batch.cpu()
                preds = logits.argmax(dim=-1)
                targets = targets
 
                for pred_prot, gt_prot in zip(out_batch.to_data_list(), batch.to_data_list()):

                    log = pred_prot.x[~gt_prot.node_mask] 
                    targ = gt_prot.y[~gt_prot.node_mask]
                    loss = crit(log, targ)
                    val_losses.append(loss.item())

                    pred = log.argmax(dim=-1)
                    val_acc_top_1.append(top_k_acc(pred, targ, 1))
                    val_acc_top_5.append(top_k_acc(pred, targ, 5))
                    val_acc_top_10.append(top_k_acc(pred, targ,10))

                batch_conf_mat = get_confusion_matrix(targets_cpu, preds, num_classes)
                global_confusion_mat += batch_conf_mat

                if hasattr(batch, "batch") and batch.batch is not None:
                    nodes_per_protein = torch.bincount(batch.batch)
                    all_positions = torch.cat([torch.arange(n.item(), device=DEVICE) for n in nodes_per_protein])
                else:
                    all_positions = torch.arange(batch.num_nodes, device=DEVICE)
                
                positions = all_positions[pred_mask].cpu()
                correct_mask = (preds == targets_cpu).long()
                
                if len(positions) > 0:
                    current_max_pos = positions.max().item() + 1
                    
                    if current_max_pos > len(pos_total):
                        pad_size = current_max_pos - len(pos_total)
                        pos_correct = torch.cat([pos_correct, torch.zeros(pad_size, dtype=torch.long)])
                        pos_total = torch.cat([pos_total, torch.zeros(pad_size, dtype=torch.long)])
                    
                    pos_total += torch.bincount(positions, minlength=len(pos_total))
                    pos_correct += torch.bincount(positions, weights=correct_mask, minlength=len(pos_correct)).long()
                        
                diag = batch_conf_mat.diag()
                recall = torch.nan_to_num(diag / batch_conf_mat.sum(dim=1), 0.0)
                precision = torch.nan_to_num(diag / batch_conf_mat.sum(dim=0), 0.0)
                f1 = torch.nan_to_num(2 * (precision * recall) / (precision + recall), 0.0)

                # Residue sequence position

                # Evaluate amino acids
                for k, amino_acid in enumerate(ResidueClassifierDataset.AMINO_ACIDS):
                     precisions[amino_acid].append(precision[k].item())
                     recalls[amino_acid].append(recall[k].item())
                     f1s[amino_acid].append(f1[k].item())

        prf_dict = {}
        precisions = {key: torch.tensor(value) for key, value in precisions.items()}
        recalls = {key: torch.tensor(value) for key, value in recalls.items()}
        f1s = {key: torch.tensor(value) for key, value in f1s.items()}
        for k, amino_acid in enumerate(ResidueClassifierDataset.AMINO_ACIDS):
            prf_dict[f"val_{amino_acid}_f1_mean"] = f1s[amino_acid].mean().item()
            prf_dict[f"val_{amino_acid}_precision_mean"] = precisions[amino_acid].mean().item()
            prf_dict[f"val_{amino_acid}_recall_mean"] = recalls[amino_acid].mean().item()
            prf_dict[f"val_{amino_acid}_f1_std"] = f1s[amino_acid].std().item()
            prf_dict[f"val_{amino_acid}_precision_std"] = precisions[amino_acid].std().item()
            prf_dict[f"val_{amino_acid}_recall_std"] = recalls[amino_acid].std().item()

        # perplexity score
        val_perplexities = torch.exp(torch.tensor(val_losses))
        prf_dict["val_perp_mean"] = val_perplexities.mean().item()
        prf_dict["val_perp_std"] = val_perplexities.std().item()
        
        val_acc_top_1 = torch.tensor(val_acc_top_1)
        val_acc_top_5 = torch.tensor(val_acc_top_5)
        val_acc_top_10 = torch.tensor(val_acc_top_10)

        prf_dict["val_top1_acc_mean"] = val_acc_top_1.mean().item()
        prf_dict["val_top5_acc_mean"] = val_acc_top_5_stats.mean().item()
        prf_dict["val_top10_acc_mean"] = val_acc_top_10_stats.mean().item()
        prf_dict["val_top1_acc_std"] = val_acc_top_1_stats.std().item()
        prf_dict["val_top5_acc_std"] = val_acc_top_5_stats.std().item()
        prf_dict["val_top10_acc_std"] = val_acc_top_10_stats.std().item()
        

        #prf_dict["rmsd_mean"] = torch.tensor(scRMSDs).mean().item()
        #prf_dict["rmsd_std_dev"] = torch.tensor(scRMSDs).std().item()

        
        # Confusion matrix
        fig, ax = plt.subplots(figsize=(12, 12))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=global_confusion_mat.numpy().astype(int),
            display_labels=ResidueClassifierDataset.AMINO_ACIDS
        )
        disp.plot(cmap='Blues', ax=ax, values_format='d')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.title("Val Set Amino Acid Confusion Matrix")

        prf_dict["val_aa_confusion_matrix"] = wandb.Image(fig)
        plt.close(fig)

        # Sequence positon accuracy
        valid_pos_mask = pos_total > 0
        if valid_pos_mask.any():
            valid_positions = torch.arange(len(pos_total))[valid_pos_mask]
            pos_accuracies = (pos_correct[valid_pos_mask] / pos_total[valid_pos_mask])

            fig_pos, ax_pos = plt.subplots(figsize=(12, 5))
            ax_pos.plot(valid_positions.numpy(), pos_accuracies, marker='.', linestyle='-', alpha=0.7)
            ax_pos.set_xlabel("Amino Acid Sequence Position (N-terminus -> C-terminus)")
            ax_pos.set_ylabel("Accuracy")
            ax_pos.set_title(f"Validation Accuracy vs. Sequence Position (Epoch {epoch+1})")
            ax_pos.grid(True, linestyle='--', alpha=0.6)
            
            prf_dict["val_position_accuracy_plot"] = wandb.Image(fig_pos)
            plt.close(fig_pos)
        
        # now let's qualitatively analyze sheaves
        if not ablate_sheaves:
            with torch.no_grad():
                for batch in tqdm(single_graph_val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):

                    data = batch.to_data_list()[0].to(DEVICE)
                    data = data.sort() 
                    _, first_sheaf, last_sheaf = model(data, return_sheaf=True)
                    first_sheaf = first_sheaf.cpu()
                    last_sheaf = last_sheaf.cpu()
                    data = data.cpu()
                    traj_id = data.traj_id if isinstance(data.traj_id, str) else data.traj_id[0]
                    _, s_1, _ = torch.linalg.svd(first_sheaf)
                    _, s_2, _ = torch.linalg.svd(last_sheaf)
                    values = s_1.numpy()
                    pos = np.arange(values.shape[1]) 
                    fig, ax = plt.subplots(figsize=(10, 4))

                    ax.violinplot(values, pos, points=60, widths=0.7, showmeans=True, showextrema=True, showmedians=True)
                    prf_dict[f"{traj_id}_singular_values_sheaf_0"] = wandb.Image(fig)
                    plt.close(fig)

                    values = s_2.numpy()
                    pos = np.arange(values.shape[1]) 
                    fig, ax = plt.subplots(figsize=(10, 4))

                    ax.violinplot(values, pos, points=60, widths=0.7, showmeans=True, showextrema=True, showmedians=True)
                    prf_dict[f"{traj_id}_singular_values_sheaf_{num_blocks-1}"] = wandb.Image(fig)
                    plt.close(fig)



        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0

        run.log(prf_dict | {"epoch_val_loss": avg_val_loss, "epoch": epoch})



    model.eval()

    global_confusion_mat = torch.zeros((num_classes, num_classes), device=DEVICE)
    test_losses = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = batch.to(DEVICE)

            batch = batch.sort() # for ConvGAT aggregation

            out_batch = model(batch)

            pred_mask = ~batch.node_mask.bool()

            test_logits = out_batch.x[pred_mask]
            test_targets = out_batch.y[pred_mask]

            test_loss = crit(test_logits, test_targets)
            test_losses.append(test_loss.item())

            preds = test_logits.argmax(dim=-1)
            batch_conf_mat = get_confusion_matrix(test_targets, preds, num_classes)
            global_confusion_mat += batch_conf_mat

    avg_test_loss = sum(test_losses) / len(test_losses) if test_losses else 0
    run.log({"final_test_loss": avg_test_loss})

    confusion_mat_cpu = global_confusion_mat.cpu()
    diag = confusion_mat_cpu.diag()

    recall = torch.nan_to_num(diag / confusion_mat_cpu.sum(dim=1), 0.0)
    precision = torch.nan_to_num(diag / confusion_mat_cpu.sum(dim=0), 0.0)
    f1 = torch.nan_to_num(2 * (precision * recall) / (precision + recall), 0.0)

    prf_dict = {}
    for k, residue in enumerate(ResidueClassifierDataset.AMINO_ACIDS):
        prf_dict[f"test_{residue}_precision"] = precision[k].item()
        prf_dict[f"test_{residue}_recall"] = recall[k].item()
        prf_dict[f"test_{residue}_f1"] = f1[k].item()

    prf_dict["test_top_1_acc"] = diag.sum().item() / confusion_mat_cpu.sum().item()
    # generate and log test confusion matrix

    # Hopefully plot will be big enough for 22 classes
    fig, ax = plt.subplots(figsize=(12, 12))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_mat_cpu.numpy().astype(int),
        display_labels=ResidueClassifierDataset.AMINO_ACIDS
    )
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.title("Test Set Confusion Matrix")

    prf_dict["test_confusion_matrix"] = wandb.Image(fig)
    run.log(prf_dict)

    plt.close(fig)

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
    args = parser.parse_args()
    train_residue_classifier(vars(args))

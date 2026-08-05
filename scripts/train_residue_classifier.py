from residue_classifier_dataset import ResidueClassifierDataset
from motion_model import MotionClassifier
from sheaf_utils import build_graph
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm import tqdm

import os
import math


def get_batch_confusion_matrix(gt_indices, pred_indices, num_classes):
    # gt_indices: shape = (B), 0 <= min(), max() < num_classes
    # pred_indices: shape = (B), 0 <= min(), max() < num_classes
    # returns: confusion_mat. shape = num_classes, num_classes

    # Convert pred indices and gt indices into one-hot encoded vectors
    # einsum "bi, bj->ij" does batched matrix multiplication. 
    # i (rows) represents Ground Truth, j (cols) represents Predictions.
    gt_one_hot = F.one_hot(gt_indices, num_classes=num_classes).float()
    pred_one_hot = F.one_hot(pred_indices, num_classes=num_classes).float()

    confusion_mat = torch.einsum("bi, bj->ij", gt_one_hot, pred_one_hot)
    return confusion_mat


def train_motion_classifier():
    # hyperparameters
    epsilon = 5.0 # in Angstroms
    learning_rate = 1e-4
    epochs = 8
    val_ratio = 0.15
    test_ratio = 0.15
    batch_size = 32 
    hidden_dim = 64
    steps = 5 # num diffusion steps
    step_size = 0.1 # diffusion step size

    # set up device 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    # load in data and create 3-way split
    df = pd.read_csv(os.path.abspath("atlas_index.csv"))
    pdb_ids = df["pdb_id"].unique()
    num_pdbs = len(pdb_ids)
    
    val_cutoff = int(val_ratio * num_pdbs)
    test_cutoff = val_cutoff + int(test_ratio * num_pdbs)
    
    val_pdbs = pdb_ids[:val_cutoff]
    test_pdbs = pdb_ids[val_cutoff:test_cutoff]
    train_pdbs = pdb_ids[test_cutoff:]
    
    train_df = df[df["motion_id"].isin(train_pdbs)]
    val_df = df[df["motion_id"].isin(val_pdbs)]
    test_df = df[df["motion_id"].isin(test_pdbs)]

    # set up wandb
    wandb.login(key=os.getenv("WANDB_KEY"))
    run = wandb.init(
        entity="jenslundsgaard7-uw-madison",
        project="SheafProtein",
        name="sheaf_diffusion_training",
        config={
            "epsilon": epsilon,
            "lr": learning_rate,
            "epochs": epochs,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "batch_size": batch_size,
            "hidden_dim": hidden_dim,
            "steps": steps,
            "step_size": step_size,
            "task":"predicting residues from motions"
        },
    )

    # WANDB artifact logging
    artifact = wandb.Artifact(name="scripts", type="model_file")
    artifact.add_file(os.path.abspath(__file__))
    
    dependencies = [ # TODO fix
        "motion_model.py", 
        "motion_classifier_dataset.py", 
        "sheaf_utils.py"
    ]
    for file in dependencies:
        if os.path.exists(file):
            artifact.add_file(os.path.abspath(file))
            
    run.log_artifact(artifact)

    # Initialize datasets
    train_dataset = ResidueClassifierDataset(train_df)
    val_dataset = ResidueClassifierDataset(val_df)
    test_dataset = ResidueClassifierDataset(test_df)

    # set up dataloaders
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=16, collate_fn=MotionClassifierDataset.pad_collate, pin_memory=True, drop_last=False) 
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=16, collate_fn=MotionClassifierDataset.pad_collate, pin_memory=True, drop_last=False) 
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, num_workers=16, collate_fn=MotionClassifierDataset.pad_collate, pin_memory=True, drop_last=False) 
    
    num_classes = len(ResidueClassifierDataset.AMINO_ACIDS)
    
    # set up new diffusion model # TODO fix all this
    F_dim = len(MotionClassifierDataset.AMINO_ACIDS) + 3
    dummy_X = torch.zeros((1, 2, 10, F_dim), device=DEVICE)
    dummy_edges = torch.zeros((1, 2, 2, 2), dtype=torch.long, device=DEVICE)

    model = MotionClassifier(
        X=dummy_X, 
        edge_index=dummy_edges, 
        num_classes=num_classes, 
        hidden_dim=hidden_dim,
        step_size = step_size,
        steps = steps
    ).to(torch.float32)
    model = model.to(DEVICE)
    
    crit = torch.nn.CrossEntropyLoss() # TODO replace this with a properly masked loss, if it exists

    # training loop
    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        model.train()
        
        for conformations1, conformations2, residues, motion_classes, lengths in pbar:
            conformations1 = conformations1.to(DEVICE) 
            conformations2 = conformations2.to(DEVICE) 
            residues = residues.to(DEVICE) 
            motion_classes = motion_classes.to(DEVICE) 
            lengths = lengths.to(DEVICE)
            
            edges, _ = build_graph(conformations1, conformations2, lengths, torch.tensor(epsilon, device=DEVICE), adjacency_matrix=False) 
            edge_index = edges.permute(0, 1, 3, 2).long()
            
            residues_one_hot = F.one_hot(residues, num_classes=len(MotionClassifierDataset.AMINO_ACIDS)) 

            center1 = conformations1.mean(dim=1, keepdim=True) 
            center2 = conformations2.mean(dim=1, keepdim=True) 

            conformations1 = (conformations1 - center1) / 10.0 
            conformations2 = (conformations2 - center2) / 10.0 

            node_features1 = torch.cat([conformations1, residues_one_hot], dim=2)  
            node_features2 = torch.cat([conformations2, residues_one_hot], dim=2) 
            
            node_features = torch.stack([node_features1, node_features2], dim=1).to(torch.float32)
            
            optimizer.zero_grad()
            logits = model(node_features, edge_index) 

            loss = crit(logits, motion_classes)
            run.log({"train_loss": loss.detach().cpu().item(), "epoch": epoch})
            
            loss.backward() 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Validation Check
        model.eval()
        val_losses = []
        with torch.no_grad():
            for conformations1, conformations2, residues, motion_classes, lengths in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                conformations1 = conformations1.to(DEVICE) 
                conformations2 = conformations2.to(DEVICE) 
                residues = residues.to(DEVICE) 
                motion_classes = motion_classes.to(DEVICE) 
                lengths = lengths.to(DEVICE)
                
                edges, _ = build_graph(conformations1, conformations2, lengths, torch.tensor(epsilon, device=DEVICE), adjacency_matrix=False) 
                edge_index = edges.permute(0, 1, 3, 2).long()
                
                residues_one_hot = F.one_hot(residues, num_classes=len(MotionClassifierDataset.AMINO_ACIDS))
                center1 = conformations1.mean(dim=1, keepdim=True) 
                center2 = conformations2.mean(dim=1, keepdim=True) 
                conformations1 = (conformations1 - center1) / 10.0 
                conformations2 = (conformations2 - center2) / 10.0 

                node_features1 = torch.cat([conformations1, residues_one_hot], dim=2)
                node_features2 = torch.cat([conformations2, residues_one_hot], dim=2) 
                
                node_features = torch.stack([node_features1, node_features2], dim=1).to(torch.float32)

                logits = model(node_features, edge_index)
                loss = crit(logits, motion_classes)
                val_losses.append(loss.cpu().item())
                
        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0
        run.log({"epoch_val_loss": avg_val_loss, "epoch": epoch})


    # Final Test Evaluation
    print("Training complete. Running final evaluation on Test Set...")
    model.eval()
    
    global_confusion_mat = torch.zeros((num_classes, num_classes), device=DEVICE)
    test_losses = []
    
    with torch.no_grad():
        for conformations1, conformations2, residues, motion_classes, lengths in tqdm(test_loader, desc="Testing"):
            conformations1 = conformations1.to(DEVICE) 
            conformations2 = conformations2.to(DEVICE) 
            residues = residues.to(DEVICE) 
            motion_classes = motion_classes.to(DEVICE) 
            lengths = lengths.to(DEVICE)
            
            edges, _ = build_graph(conformations1, conformations2, lengths, torch.tensor(epsilon, device=DEVICE), adjacency_matrix=False) 
            edge_index = edges.permute(0, 1, 3, 2).long()
            
            residues_one_hot = F.one_hot(residues, num_classes=len(MotionClassifierDataset.AMINO_ACIDS))
            center1 = conformations1.mean(dim=1, keepdim=True) 
            center2 = conformations2.mean(dim=1, keepdim=True) 
            conformations1 = (conformations1 - center1) / 10.0 
            conformations2 = (conformations2 - center2) / 10.0 

            node_features1 = torch.cat([conformations1, residues_one_hot], dim=2) 
            node_features2 = torch.cat([conformations2, residues_one_hot], dim=2) 
            
            node_features = torch.stack([node_features1, node_features2], dim=1).to(torch.float32)

            logits = model(node_features, edge_index)

            # track test loss
            test_loss = crit(logits, motion_classes)
            test_losses.append(test_loss.cpu().item())
            
            preds = logits.argmax(dim=-1)
            batch_conf_mat = get_batch_confusion_matrix(motion_classes, preds, num_classes)
            global_confusion_mat += batch_conf_mat

    avg_test_loss = sum(test_losses) / len(test_losses) if test_losses else 0
    run.log({"final_test_loss": avg_test_loss})

    # Calculate metrics based on entire test set
    confusion_mat_cpu = global_confusion_mat.cpu()
    diag = confusion_mat_cpu.diag()
    
    recall = torch.nan_to_num(diag / confusion_mat_cpu.sum(dim=1), 0.0)
    precision = torch.nan_to_num(diag / confusion_mat_cpu.sum(dim=0), 0.0)
    f1 = torch.nan_to_num(2 * (precision * recall) / (precision + recall), 0.0)

    prf_dict = {}
    for k, motion_class in enumerate(MotionClassifierDataset.MOTION_CLASSES):
        prf_dict[f"test_{motion_class}_precision"] = precision[k].item()
        prf_dict[f"test_{motion_class}_recall"] = recall[k].item()
        prf_dict[f"test_{motion_class}_f1"] = f1[k].item()

    # generate and log test confusion matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_mat_cpu.numpy().astype(int), 
        display_labels=MotionClassifierDataset.MOTION_CLASSES
    )
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right') 
    plt.title("Test Set Confusion Matrix")

    prf_dict["test_confusion_matrix"] = wandb.Image(fig)
    run.log(prf_dict) 

    plt.close(fig)
    
    run.finish()

if __name__ == "__main__":
    train_motion_classifier()

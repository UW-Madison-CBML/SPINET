# for use them directly
from motion_classifier_dataset import MotionClassifierDataset
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
    val_ratio = 0.3 
    batch_size = 32 
    hidden_dim = 64
    diffusion_steps = 5
    step_size = 0.1 # diffusion step size

    
    # set up device 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    # load in data
    df = pd.read_csv(os.path.abspath("motions.csv"))
    motion_ids = df["motion_id"].unique()
    val_motions = motion_ids[:int(val_ratio * len(motion_ids))]
    df_mask = df["motion_id"].isin(val_motions)

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
            "batch_size": batch_size,
            "hidden_dim": hidden_dim,
            "steps": diffusion_steps,
            "step_size": step_size
        },
    )

    # WANDB artifact logging
    artifact = wandb.Artifact(name="scripts", type="model_file")
    
    # Use __file__ to dynamically get the path of this current training script
    artifact.add_file(os.path.abspath(__file__))
    
    # Add your other dependencies based on your imports
    dependencies = [
        "motion_model.py", 
        "motion_classifier_dataset.py", 
        "sheaf_utils.py"
    ]
    for file in dependencies:
        if os.path.exists(file):
            artifact.add_file(os.path.abspath(file))
            
    run.log_artifact(artifact)

    # set up validation split
    val_df = df[df_mask]
    df = df[~ df_mask]
    dataset = MotionClassifierDataset(df)
    val_dataset = MotionClassifierDataset(val_df)

    # set up dataloader
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=16, collate_fn=MotionClassifierDataset.pad_collate, pin_memory=True, drop_last=False) 
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=16, collate_fn=MotionClassifierDataset.pad_collate, pin_memory=True, drop_last=False) 
    
    num_classes = len(MotionClassifierDataset.MOTION_CLASSES)
    
    # set up new diffusion model
    # Generate dummy data to initialize the model's dynamically tracked dimensions
    F_dim = len(MotionClassifierDataset.AMINO_ACIDS) + 3
    dummy_X = torch.zeros((1, 2, 10, F_dim), device=DEVICE)
    dummy_edges = torch.zeros((1, 2, 2, 2), dtype=torch.long, device=DEVICE)

    model = MotionClassifier(
        X=dummy_X, 
        edge_index=dummy_edges, 
        num_classes=num_classes, 
        hidden_dim=hidden_dim,

    ).to(torch.float32)
    model = model.to(DEVICE)
    
    # set up other training stuff
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5) 
    crit = torch.nn.CrossEntropyLoss()

    # training  
    #TODO: Adapt build_graph for COO format. Edge_lengths unnecessary, and should output PyG's expected 2,E format directly.
    for epoch in range(epochs):
        pbar = tqdm(loader)
        model = model.train()
        for conformations1, conformations2, residues, motion_classes, lengths in pbar:

            conformations1 = conformations1.to(DEVICE) # B, T, 3  
            conformations2 = conformations2.to(DEVICE) # B, T, 3 

            residues = residues.to(DEVICE) # B, T
            motion_classes = motion_classes.to(DEVICE) # B

            lengths = lengths.to(DEVICE)
            
            # Extract edges without adjacency matrix (B, 2, E, 2)
            edges, _ = build_graph(conformations1, conformations2, lengths, torch.tensor(epsilon, device=DEVICE), adjacency_matrix=False) 
            
            # Reshape edges from (B, 2, E, 2) to PyG's expected (B, 2, 2, E) format
            edge_index = edges.permute(0, 1, 3, 2).long()
            
            residues_one_hot = F.one_hot(residues, num_classes=len(MotionClassifierDataset.AMINO_ACIDS)) # B, T, amino_acids

            # center the conformations to origin
            #TODO: is this necessary?
            center1 = conformations1.mean(dim=1, keepdim=True) # B, 1, 3
            center2 = conformations2.mean(dim=1, keepdim=True) # B, 1, 3

            # Downscale to prevent param explosion
            conformations1 = (conformations1 - center1) / 10.0 # B, T, 3
            conformations2 = (conformations2 - center2) / 10.0 # B, T, 3

            node_features1 = torch.cat([conformations1, residues_one_hot], dim=2) # B, T, 3 + amino_acids 
            node_features2 = torch.cat([conformations2, residues_one_hot], dim=2) 
            
            node_features = torch.stack([node_features1, node_features2], dim=1)
            node_features = node_features.to(torch.float32)
            
            optimizer.zero_grad()

            # Forward pass through the sheaf diffusion model
            logits = model(node_features, edge_index) 

            # compare prediction to ground truth classes
            loss = crit(logits, motion_classes)
            run.log({"loss": loss.detach().cpu().item()})
           
            # back propagate and reset
            loss.backward() 

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

        # validation
        model.eval()
        
        # Accumulator for global confusion matrix
        global_confusion_mat = torch.zeros((num_classes, num_classes), device=DEVICE)
        
        with torch.no_grad():
            for conformations1, conformations2, residues, motion_classes, lengths in val_loader:
                conformations1 = conformations1.to(DEVICE) # B, T, 3  
                conformations2 = conformations2.to(DEVICE) # B, T, 3 

                residues = residues.to(DEVICE) # B, T
                motion_classes = motion_classes.to(DEVICE) # B

                lengths = lengths.to(DEVICE)
                
                edges, _ = build_graph(conformations1, conformations2, lengths, torch.tensor(epsilon, device=DEVICE), adjacency_matrix=False) 
                edge_index = edges.permute(0, 1, 3, 2).long()
                
                residues_one_hot = F.one_hot(residues, num_classes=len(MotionClassifierDataset.AMINO_ACIDS)) # B, T, amino_acids
                node_features1 = torch.cat([conformations1, residues_one_hot], dim=2) # B, T, 3 + amino_acids 
                node_features2 = torch.cat([conformations2, residues_one_hot], dim=2) 
                
                node_features = torch.stack([node_features1, node_features2], dim=1)
                node_features = node_features.to(torch.float32)

                logits = model(node_features, edge_index)

                # compare prediction to ground truth classes
                loss = crit(logits, motion_classes)
                run.log({"val_loss": loss.cpu().item()})
                
                preds = logits.argmax(dim=-1)
                
                # compute batch confusion matrix and accumulate
                batch_conf_mat = get_batch_confusion_matrix(motion_classes, preds, num_classes)
                global_confusion_mat += batch_conf_mat

        # Calculate metrics globally based on entire validation set
        confusion_mat_cpu = global_confusion_mat.cpu()
        diag = confusion_mat_cpu.diag()
        
        # sum(dim=1) gives total actual instances of each class
        recall = torch.nan_to_num(diag / confusion_mat_cpu.sum(dim=1), 0.0)
        
        # sum(dim=0) gives total predicted instances of each class
        precision = torch.nan_to_num(diag / confusion_mat_cpu.sum(dim=0), 0.0)
        
        f1 = torch.nan_to_num(2 * (precision * recall) / (precision + recall), 0.0)

        prf_dict = {}
        for k, motion_class in enumerate(MotionClassifierDataset.MOTION_CLASSES):
            prf_dict[f"{motion_class}_precision"] = precision[k].item()
            prf_dict[f"{motion_class}_recall"] = recall[k].item()
            prf_dict[f"{motion_class}_f1"] = f1[k].item()

        # do display for the confusion matrix 
        fig, ax = plt.subplots(figsize=(10, 10))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion_mat_cpu.numpy().astype(int), 
            display_labels=MotionClassifierDataset.MOTION_CLASSES
        )
        disp.plot(cmap='Blues', ax=ax, values_format='d')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right') 

        prf_dict["confusion_matrix"] = wandb.Image(fig)
        prf_dict["epoch"] = epoch
        run.log(prf_dict) 

        plt.close(fig)
 
    run.finish()

if __name__ == "__main__":
    train_motion_classifier()

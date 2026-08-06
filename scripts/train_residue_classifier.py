import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm import tqdm

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from residue_classifier_dataset import ResidueClassifierDataset
from residue_model import NodeSheafGATClassifier
from sheaf_utils import build_graph


def get_confusion_matrix(gt_indices, pred_indices, num_classes):
    """Compute confusion matrix over 1D array of pred and target."""
    gt_one_hot = F.one_hot(gt_indices, num_classes=num_classes).float()
    pred_one_hot = F.one_hot(pred_indices, num_classes=num_classes).float()

    confusion_mat = torch.einsum("bi, bj->ij", gt_one_hot, pred_one_hot)
    return confusion_mat


def train_residue_classifier():
    # hyperparameters
    epsilon = 5.0 # in Angstroms
    learning_rate = 1e-4
    epochs = 8
    val_ratio = 0.15
    test_ratio = 0.15
    batch_size = 32 
    hidden_dim = 64
    num_blocks = 4
    num_heads = 4

    # set up device 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    # load in data and create 3-way split
    # the data is split into different csvs
    df = pd.concat([pd.read_csv(os.path.join("md_data", f"atlas_index_{i}.csv")) for i in range(4)], axis=0, ignore_index=True)
    pdb_ids = df["pdb_id"].unique()
    num_pdbs = len(pdb_ids)
    
    val_cutoff = int(val_ratio * num_pdbs)
    test_cutoff = val_cutoff + int(test_ratio * num_pdbs)
    
    val_pdbs = pdb_ids[:val_cutoff]
    test_pdbs = pdb_ids[val_cutoff:test_cutoff]
    train_pdbs = pdb_ids[test_cutoff:]
    
    train_df = df[df["traj_id"].isin(train_pdbs)]
    val_df = df[df["traj_id"].isin(val_pdbs)]
    test_df = df[df["traj_id"].isin(test_pdbs)]

    # set up wandb
    wandb.login(key=os.getenv("WANDB_KEY"))
    run = wandb.init(
        entity="jenslundsgaard7-uw-madison",
        project="SheafProtein",
        name="residue_classifier_training",
        config={
            "epsilon": epsilon,
            "lr": learning_rate,
            "epochs": epochs,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "batch_size": batch_size,
            "hidden_dim": hidden_dim,
            "task":"predicting residues from motions"
        },
    )

    # WANDB artifact logging
    artifact = wandb.Artifact(name="scripts", type="model_file")
    artifact.add_file(os.path.abspath(__file__))
    
    dependencies = [ # TODO fix
        "residue_model.py", 
        "residue_classifier_dataset.py", 
        "sheaf_utils.py"
    ]
    for file in dependencies:
        if os.path.exists(file):
            artifact.add_file(os.path.abspath(file))
            
    run.log_artifact(artifact)

    # Initialize datasets
    train_dataset = ResidueClassifierDataset(train_df, epsilon=epsilon)
    val_dataset = ResidueClassifierDataset(val_df, epsilon=epsilon)
    test_dataset = ResidueClassifierDataset(test_df, epsilon=epsilon)

    # set up dataloaders
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=16, collate_fn=ResidueClassifierDataset.pad_collate, pin_memory=True, drop_last=False) 
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=16, collate_fn=ResidueClassifierDataset.pad_collate, pin_memory=True, drop_last=False) 
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, num_workers=16, collate_fn=ResidueClassifierDataset.pad_collate, pin_memory=True, drop_last=False) 
    
    num_classes = len(ResidueClassifierDataset.AMINO_ACIDS)
    
    # set up new diffusion model # TODO fix all this
    # do we need this dummy data initialization?
    # ---------------------------------------------

    model = NodeSheafGATClassifier(
        num_classes=num_classes, 
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        ablate_sheaves=False
    ).to(DEVICE)

    # -----------------------------------------
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    crit = torch.nn.CrossEntropyLoss() # TODO replace this with a properly masked loss, if it exists

    # training loop
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
         
        for batch in pbar:
            batch = batch.to(DEVICE)

            optimizer.zero_grad()

            out_batch = model(batch)

            pred_mask = ~batch.node_mask.bool()

            loss = crit(out_batch.x[pred_mask], batch.y[pred_mask])
            run.log({"train_loss": loss.item(), "epoch": epoch})

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Validation Check
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                batch = batch.to(DEVICE)
                out_batch = model(batch)

                pred_mask = ~batch.node_mask.bool()
                loss = crit(out_batch.x[pred_mask], batch.y[pred_mask])

                val_losses.append(loss.item())
               
        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0
        run.log({"epoch_val_loss": avg_val_loss, "epoch": epoch})


    # Final Test Evaluation
    print("Training complete. Running final evaluation on Test Set...")
    model.eval()
    
    global_confusion_mat = torch.zeros((num_classes, num_classes), device=DEVICE)
    test_losses = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = batch.to(DEVICE) 
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

    # Calculate metrics based on entire test set
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
    train_residue_classifier()

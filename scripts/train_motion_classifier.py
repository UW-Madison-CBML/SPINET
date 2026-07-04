# for use them directly
from motion_classifier_dataset import MotionClassifierDataset
from motion_model import SheafMotionClassifier
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

def prfcm(gt_indices, pred_indices, num_classes):
    # gt_indicies1: shape = (B), 0 <= min(), max() < num_classes
    # pred_indicies2: shape = (B), 0 <= min(), max() < num_classes
    # 0 <= i < num_classes
    # returns: precision, recall, f1. shape = num_classes
    #          confusion_mat. shape = num_classes, num_classes

    confusion_mat = torch.einsum("bi, bj->ij", F.one_hot(pred_indices, num_classes=num_classes), F.one_hot(gt_indices, num_classes=num_classes))
    diag = confusion_mat[torch.arange(num_classes),torch.arange(num_classes)]
    recall = torch.nan_to_num(diag/confusion_mat.sum(dim=0), 0.0)
    precision = torch.nan_to_num(diag/confusion_mat.sum(dim=1), 0.0)
    f1 = torch.nan_to_num(2 * (precision * recall) / (precision + recall), 0.0)
    return recall, precision, f1, confusion_mat

   



def train_motion_classifier():
    # hyperparameters
    epsilon = 5.0 # in Angstroms
    learning_rate = 1e-4
    epochs = 8
    val_ratio = 0.3 
    use_adjacency_mat = True
    batch_size = 4
    
    # set up device 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    # load in data
    df = pd.read_csv(os.path.abspath("motions.csv"))
    motion_ids = df["motion_id"].unique()
    #np.random.shuffle(motion_ids) # optionally shuffle the motion ids
    val_motions = motion_ids[:int(val_ratio * len(motion_ids))]
    df_mask = df["motion_id"].isin(val_motions)

    #set up wandb
    wandb.login(key=os.getenv("WANDB_KEY"))
    run = wandb.init(
        entity="jenslundsgaard7-uw-madison",
        project="SheafProtein",
        name="sheaf_training",
        config={
            "epsilon":epsilon,
            "lr":learning_rate,
            "epochs":epochs,
            "val_motions": val_motions,
            "val_ratio":val_ratio,
            "use_adjacency_mat":use_adjacency_mat,
            "batch_size":batch_size
        },

    )
    # set up validation split
    val_df = df[df_mask]
    df = df[~ df_mask]
    dataset = MotionClassifierDataset(df)
    val_dataset = MotionClassifierDataset(val_df)

    # set up dataloader
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=16, collate_fn=MotionClassifierDataset.pad_collate, pin_memory=True, drop_last=False) 
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=16, collate_fn=MotionClassifierDataset.pad_collate, pin_memory=True, drop_last=False) 
    
    # model.__init__(self, node_features, stalk_dimensions,lstm_hidden_dim=8, num_classes=5, hidden_dim=64, adjacency_matrix=True):
    #set up model
    model = SheafMotionClassifier(len(MotionClassifierDataset.AMINO_ACIDS)+3, 2, lstm_hidden_dim=2, hidden_dim=8, num_classes=len(MotionClassifierDataset.MOTION_CLASSES), adjacency_matrix=use_adjacency_mat)
    model = model.to(DEVICE)
    
    # set up other training stuff
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5) 
    crit = torch.nn.CrossEntropyLoss()
     
    # training  
    for epoch in range(epochs):
        pbar = tqdm(loader)
        for conformations1, conformations2, residues, motion_classes, lengths in pbar:

            conformations1 = conformations1.to(DEVICE) # B, T, 3  
            conformations2 = conformations2.to(DEVICE) # B, T, 3 
            residues = residues.to(DEVICE) # B, T
            motion_classes = motion_classes.to(DEVICE) # B
            if(use_adjacency_mat):
                mats = build_graph(conformations1, conformations2, lengths, torch.tensor(epsilon, device=DEVICE), adjacency_matrix=True) # B, 2, T, T, type=bool
                print("mats.shape: ", mats.shape)    

            else:
                edges, edge_lengths = build_graph(conformations1, conformations2, lengths, torch.tensor(epsilon, device=DEVICE), adjacency_matrix=False) # B, 2, E, 2,type= unsigned int
            residues_one_hot = F.one_hot(residues, num_classes=len(MotionClassifierDataset.AMINO_ACIDS)) # B, T, amino_acids
            node_features1 = torch.cat([conformations1, residues_one_hot],dim=2) # B, T, 3 + amino_acids 
            node_features2 = torch.cat([conformations2, residues_one_hot],dim=2) 
            
            node_features = torch.stack([node_features1, node_features2], dim=1)
            node_features = node_features.to(torch.float)
            # just in case any previous operations have been accumulating gradients
            optimizer.zero_grad()

            if use_adjacency_mat:
                logits = model(node_features, lengths, matrix=mats) # B 
            else:
                logits = model(node_features, lengths, edges=edges, edge_lengths = edge_lengths) # B 

            # compare prediction to ground truth classes
            loss = crit(logits, motion_classes)
            run.log({"loss":loss.detach().cpu().item()})
           
            # back propagate and reset
            loss.backward() 
            optimizer.step()

        
        # validation
        model.eval()
        prfcm = [] 
        # confusion mat for summing
        confusion_mat = np.zeros((len(MotionClassifierDataset.MOTION_CLASSES), len(MotionClassifierDataset.MOTION_CLASSES)), dtype=int)
        with torch.no_grad():
            for conformations1, conformations2, residues, motion_classes, lengths in val_loader:
                conformations1 = conformations1.to(DEVICE) # B, T, 3  
                conformations2 = conformations2.to(DEVICE) # B, T, 3 

                residues = residues.to(DEVICE) # B, T
                motion_classes = motion_classes.to(DEVICE) # B
                if(use_adjacency_mat):
                    mats = build_graph(conformations1, conformations2, lengths, torch.tensor(epsilon, device=DEVICE), adjacency_matrix=True) # B, 2, E, 2
                else:
                    edges, edge_paddings = build_graph(conformations1, conformations2, lengths, torch.tensor(epsilon, device=DEVICE), adjacency_matrix=False) # B, 2, E, 2
                residues_one_hot = F.one_hot(residues, num_classes=len(MotionClassifierDataset.AMINO_ACIDS)) # B, T, amino_acids
                node_features1 = torch.cat([conformations1, residues_one_hot],dim=2) # B, T, 3 + amino_acids 
                node_features2 = torch.cat([conformations2, residues_one_hot],dim=2) 
                
                node_features = torch.stack([node_features1, node_features2], dim=1)

                if use_adjacency_mat:
                    logits = model(node_features, lengths, matrix=mats) # B 
                else:
                    logits = model(node_features, lengths, edges=edges, edge_lengths = edge_lengths) # B 

                # compare prediction to ground truth classes
                loss = crit(logits, motion_classes)
                run.log({"val_loss":loss.cpu().item()})
                preds = logits.cpu().argmax(dim=-1)
                gt = motion_classes.cpu()
                # get metrics and confusion mat 
                # metrics = prec, rec, f1, confusion
                metrics = prfcm(gt, preds, len(MotionClassifierDataset.MOTION_CLASSES))
                # sum confusion mat
                confusion_mat += metrics[3].numpy().astype(int)
                # add rpf metrics 
                prf.append(torch.stack(metrics[:3], dim=0))
        # do a bunch of logging 
        prf = torch.stack(prf, dim=0) # len(val_loader), 3, num_classes
        prf = torch.stack([prf.mean(dim=0), prf.std(dim=0)],dim=0)
        prf_dict = {}
        for i,agg in enumerate(["", "_std"]):
            for j, metric in enumerate(["precision","recall","f1"]):
                for k, motion_class in enumerate(MotionClassifierDataset.MOTION_CLASSES):
                    prf_dict[f"{motion_class}_{metric}{agg}"] = prfcm[i,j,k]
        run.log(prf_dict)   

        # do display for the confusion matrix 
        fig, ax = plt.subplots(figsize=(10, 10))
        disp = ConfusionMatrixDisplay(confusion_matrix = confusion_mat, display_labels = MotionClassifierDataset.MOTION_CLASSES)
        disp.plot(cmap='Blues', ax=ax, values_format='d')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right') 
        run.log({"confusion_matrix": wandb.Image(fig)}) 
        plt.close(fig)
 


    

if __name__ == "__main__":
    train_motion_classifier()

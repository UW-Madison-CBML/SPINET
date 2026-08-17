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

def get_confusion_matrix(gt_indices, pred_indices, num_classes):
    """Compute confusion matrix over 1D array of pred and target."""
    gt_one_hot = F.one_hot(gt_indices, num_classes=num_classes).float()
    pred_one_hot = F.one_hot(pred_indices, num_classes=num_classes).float()

    confusion_mat = torch.einsum("bi, bj->ij", gt_one_hot, pred_one_hot)
    return confusion_mat


def train_residue_classifier(args_dict):
    # hyperparameters
    epsilon = 5.0 # in Angstroms
    learning_rate = 1e-3
    epochs = 8
    val_ratio = 0.15
    test_ratio = 0.15
    batch_size = 32
    hidden_dim = 16
    stalk_dim = 8
    num_blocks = 8
    num_heads = 8
    masking_ratio = 0.75
    ablate_sheaves=args_dict["ablate_sheaves"]
    run_name = args_dict["run_name"]
    restriction_map_type=args_dict["restriction_map_type"]
    seed=42
    num_timesteps = 32
    use_scheduler=False
    
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch_rng = torch.Generator(); torch_rng = torch_rng.manual_seed(seed)
    np_rng = np.random.default_rng(seed=seed)

    # CAUTION: this is based on the order of features defined in load_dynamics.py and is used to label the columns of the npy features file and is liable to change
    # ensure load_dynamics is correctly implemented w.r.t the below
    FEATURE_COLUMNS=["x","y","z","dx","dy","dz","bond_len", "bond_ang"]

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
    train_dataset = ResidueClassifierDataset(train_df, epsilon=epsilon, fixed_length=num_timesteps, traj_len=200)
    val_dataset = ResidueClassifierDataset(val_df, epsilon=epsilon, fixed_length=num_timesteps, traj_len=200)
    test_dataset = ResidueClassifierDataset(test_df, epsilon=epsilon, fixed_length=num_timesteps, traj_len=200)

    # set up dataloaders
    train_loader = DataLoader(train_dataset, shuffle=True, generator=torch_rng, batch_size=batch_size, num_workers=16, persistent_workers=True, worker_init_fn=ResidueClassifierDataset.worker_init_fn, collate_fn=lambda batch:train_dataset.graph_collate(batch), pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=16, collate_fn=lambda batch:val_dataset.graph_collate(batch), pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, num_workers=16, collate_fn=lambda batch:test_dataset.graph_collate(batch), pin_memory=True, drop_last=False)


    num_classes = len(ResidueClassifierDataset.AMINO_ACIDS)

    # set up new diffusion model # TODO fix all this
    # do we need this dummy data initialization?
    # ---------------------------------------------

    model = NodeSheafClassifier(
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        stalk_dim=stalk_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        ablate_sheaves=ablate_sheaves,
        num_timesteps=num_timesteps,
        restriction_map_type=restriction_map_type
    ).to(DEVICE)

    # -----------------------------------------

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    crit = torch.nn.CrossEntropyLoss() # TODO replace this with a properly masked loss, if it exists
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, len(train_loader))

    # training loop
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for batch in pbar:
            batch = batch.to(DEVICE)

            batch = batch.sort()
            optimizer.zero_grad()

            out_batch = model(batch)

            pred_mask = ~batch.node_mask.bool()

            loss = crit(out_batch.x[pred_mask], batch.y[pred_mask])
            run.log({"train_loss": loss.item(), "epoch": epoch})

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if use_scheduler:
                scheduler.step()

        # Validation Check
        model.eval()
        val_losses = []

        global_confusion_mat = torch.zeros((num_classes, num_classes))
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                batch = batch.to(DEVICE)
                batch = batch.sort() 
                out_batch = model(batch)

                pred_mask = ~batch.node_mask.bool()
                loss = crit(out_batch.x[pred_mask], batch.y[pred_mask])
                logits = out_batch.x[pred_mask].cpu()
                targets = out_batch.y[pred_mask].cpu()

                val_losses.append(loss.item())
                preds = logits.argmax(dim=-1)
                batch_conf_mat = get_confusion_matrix(targets, preds, num_classes)
                global_confusion_mat += batch_conf_mat

        
        diag = global_confusion_mat.diag()
        recall = torch.nan_to_num(diag / global_confusion_mat.sum(dim=1), 0.0)
        precision = torch.nan_to_num(diag / global_confusion_mat.sum(dim=0), 0.0)
        f1 = torch.nan_to_num(2 * (precision * recall) / (precision + recall), 0.0)

        prf_dict = {}
        for k, residue in enumerate(ResidueClassifierDataset.AMINO_ACIDS):
            prf_dict[f"val_{residue}_precision"] = precision[k].item()
            prf_dict[f"val_{residue}_recall"] = recall[k].item()
            prf_dict[f"val_{residue}_f1"] = f1[k].item()
        prf_dict["val_top_1_acc"] = diag.sum().item() / global_confusion_mat.sum().item()

        fig, ax = plt.subplots(figsize=(12, 12))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=global_confusion_mat.numpy().astype(int),
            display_labels=ResidueClassifierDataset.AMINO_ACIDS
        )
        disp.plot(cmap='Blues', ax=ax, values_format='d')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.title("Val Set Confusion Matrix")

        prf_dict["val_confusion_matrix"] = wandb.Image(fig)

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
    parser.add_argument('--restriction-map-type', type=str, default="low_rank", choices=['low_rank', 'orthogonal', 'arbitrary'])
    args = parser.parse_args()
    train_residue_classifier(vars(args))

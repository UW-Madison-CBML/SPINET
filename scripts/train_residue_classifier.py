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
from invariant_features_sheaf_model import NodeSheafClassifier, NodeOnlyMLP
import itertools
from itertools import product
from contextlib import nullcontext
from load_dynamics import BACKBONE_ATOMS
from Bio.SeqUtils import seq1
from alphafold.common import residue_constants
import time
import json
import subprocess
from scrmsd import load_esmfold, fold_sequences, evaluate_batch_rmsd
from huggingface_hub import login, HfApi, hf_hub_url, hf_hub_download, create_repo
from stats_utils import get_confusion_matrix, top_k_acc
from sheaf_utils import sheaf_laplacian
from torch.profiler import profile, ProfilerActivity, record_function
import h5py
from rgnn import RGNN
from torch.utils.data import BatchSampler


class ProteinSampler(BatchSampler):
    def __init__(self, dataset, num_nodes, rng=None):
        super().__init__(None, 1,False)
        self.dataset = dataset
        self.num_nodes = num_nodes
        self.batches = []
        self.rng = rng
        ds_index_idx = 0
        if self.rng is not None:
            self.index = torch.randperm(len(self.dataset.index), generator=self.rng).tolist()
        else:
            self.index = list(range(len(self.dataset.index)))



        assert all(len(tup) == 3 for tup in self.dataset.index), "index does not contain 3 values"
        with h5py.File(self.dataset.h5_path, "r") as f:
            while(ds_index_idx < len(dataset.index)):
                batch = []
                batch_num_nodes = 0
                while( ds_index_idx < len(dataset.index)): # loop while adding to a batch. if we we run out of ds_index_idxs we just end the batch, or if we get big enough
                    group_idx, _, _ = self.dataset.index[self.index[ds_index_idx]]
                    traj_id = self.dataset.groups[group_idx]
                    num_res = f[traj_id]["coordinates"].shape[0]
                    if num_res > self.num_nodes: # just discard pdb id if too big
                        ds_index_idx += 1
                    elif batch_num_nodes + num_res <= self.num_nodes: # first make sure it even fits
                        batch.append(self.index[ds_index_idx])
                        batch_num_nodes += num_res
                        ds_index_idx += 1
                    else: 
                        break
                if len(batch) > 0:
                    self.batches.append(batch)



    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def load_df_from_pdbs(local_path, file_name_format="p_c-t"):
    files = [path for path in os.listdir() if path.endswith(".pdb")] 
    
def run_val(run, model, loader, dataset, epoch, device, crit, esmfold_tokenizer, esmfold_model, val_name="val", num_classes = len(ResidueClassifierDataset.AMINO_ACIDS), cm_title="Amino Acid Confusion Matrix", test_val=False):
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
    pred_seqs = []
    pred_pdbs = []

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
            data_list = out_batch.to_data_list() # list of B per-protein Data objects (model output)
            gt_list = batch.to_data_list()  # list of B per-protein Data objects (ground truth)

            for i, gt_data in enumerate(gt_list):
                pred_data = data_list[i]
                pred_pdbs.append(gt_data.traj_id if isinstance(gt_data.traj_id, str) else gt_data.traj_id[0])

                pred_idx = pred_data.x.argmax(dim=-1) # (R_i,) predicted class index per residue, protein i (R_i = residue count, varies per protein)

                graph_mask = ~gt_data.node_mask.bool()  # (R_i,) True at masked (prediction-target) residues

                pred_idx[~graph_mask] = gt_data.y[~graph_mask]  # unmasked residues get their known ground-truth identity instead of the prediction

                seq_str = "".join([seq1(ResidueClassifierDataset.AMINO_ACIDS[idx.item()]) for idx in pred_idx])  # length-R_i amino-acid string
                pred_seqs.append(seq_str )  # accumulates to B entries, insertion order == gt_list order

            """# Get gt coordinates from trajectory
            traj_tensors = [prot.pos for prot in gt_list]

            lengths = torch.tensor([traj_tensor.shape[0] for traj_tensor in traj_tensors])  # (B,) residue count R_i per protein
            print(lengths)
            pad_size = lengths.max().item() 

            padded_tensors = [F.pad(traj_tensor, (0,0, 0,0, 0,0, 0,pad_size-traj_tensor.shape[0]), mode="constant", value=0.0) for traj_tensor in traj_tensors]

            backbone_tensor = torch.stack(padded_tensors, dim=0).permute(0,2,1,3,4)          # (B, T, num_res_padded, num_atoms, 3)
            gt_seq_mask = (lengths[:,None] > torch.arange(pad_size)[None,:])[:, None, :].repeat(1, backbone_tensor.shape[1], 1)  # (B, T, pad_size) bool

            rmsd = evaluate_batch_rmsd(pred_seqs, backbone_tensor, gt_seq_mask, esmfold_tokenizer, esmfold_model, device=device)
            scrmsd.extend(rmsd.tolist())"""

            out_batch = out_batch.cpu()
            batch = batch.cpu()
            preds = logits.argmax(dim=-1).cpu() 
            targets_cpu = targets.cpu()


            for pred_prot, gt_prot in zip(data_list, gt_list):

                pred_mask = ~gt_prot.node_mask.bool()

                log = pred_prot.x[pred_mask]
                targ = gt_prot.y[pred_mask]     
                loss = crit(log, targ)      
                losses.append(loss.item())

                acc_top_1.append(top_k_acc(log, targ, 1))
                acc_top_5.append(top_k_acc(log, targ, 5))
                acc_top_10.append(top_k_acc(log, targ,10))

            batch_conf_mat = get_confusion_matrix(targets_cpu, preds, num_classes)  
            global_confusion_mat += batch_conf_mat

            diag = batch_conf_mat.diag()                                    
            recall = torch.nan_to_num(diag / batch_conf_mat.sum(dim=1), 0.0)      
            precision = torch.nan_to_num(diag / batch_conf_mat.sum(dim=0), 0.0)   
            f1 = torch.nan_to_num(2 * (precision * recall) / (precision + recall), 0.0)  


            for k, amino_acid in enumerate(ResidueClassifierDataset.AMINO_ACIDS):
                 precisions[amino_acid].append(precision[k].item())
                 recalls[amino_acid].append(recall[k].item())
                 f1s[amino_acid].append(f1[k].item())




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
    pred_string_df = pd.DataFrame({"pdb":pred_pdbs, "seq":pred_seqs})
    prf_dict["pred_seqs"] = wandb.Table(dataframe=pred_string_df)

    # perplexity score
    perplexities = torch.exp(torch.tensor(losses))  # losses/perplexities: (total_proteins,) -- one scalar per protein, accumulated across every batch in the loader
    prf_dict[f"{val_name}_perp_mean"] = (pm := perplexities.mean().item())
    prf_dict[f"{val_name}_perp_std"] = (ps := perplexities.std().item())

    #scrmsd = torch.tensor(scrmsd)
    #prf_dict[f"{val_name}_rmsd_mean"] = scrmsd.mean().item()
    #prf_dict[f"{val_name}_rmsd_std"] = scrmsd.std().item()

    acc_top_1 = torch.tensor(acc_top_1)
    acc_top_5 = torch.tensor(acc_top_5)
    acc_top_10 = torch.tensor(acc_top_10)

    prf_dict[f"{val_name}_top1_acc_mean"] = (a1m := acc_top_1.mean().item())
    prf_dict[f"{val_name}_top5_acc_mean"] = (a5m := acc_top_5.mean().item())
    prf_dict[f"{val_name}_top10_acc_mean"] = (a10m := acc_top_10.mean().item())
    prf_dict[f"{val_name}_top1_acc_std"] = (a1s := acc_top_1.std().item())
    prf_dict[f"{val_name}_top5_acc_std"] = (a5s := acc_top_5.std().item())
    prf_dict[f"{val_name}_top10_acc_std"] = (a10s := acc_top_10.std().item())

    print(f"{sum(p.numel() for p in model.parameters() if p.requires_grad)} & ${a1m:.3f} \\pm {a1s:.3f}$ & ${a5m:.3f} \\pm {a5s:.3f}$ & ${a10m:.3f} \\pm {a10s:.3f}$ & ${pm:.3f} \\pm {ps:.3f}$") # & ${rmsdM: .3f} \\pm {rmsdS: .3f}$") 

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

    # initiate img_dict
    img_dict = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Loading Sheaves", leave=False):

            data = batch.to_data_list()[0].to(device)
            data = data.sort() 
            _, first_sheaf, last_sheaf = model(data, return_sheaf=True)
            _,d,_ = first_sheaf.shape
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
            last_eigvals = last_eigs.eigenvalues.real
            last_eigvecs = last_eigs.eigenvectors.real

            # work with first sheaf
            largest_eigval_idx = first_eigvals.argmax().item()
            largest_eigvec = first_eigvecs[largest_eigval_idx]

            # need to be careful with reshape
            per_node_signals = largest_eigvec.reshape(-1, d)
            covariance_mat = F.cosine_similarity(*torch.broadcast_tensors(per_node_signals[:,None,:], per_node_signals[None,:,:]), dim=-1)
            fig, ax = plt.subplots()
            ax.imshow(covariance_mat)
            ax.set_title(f"{traj_id}")
            img_dict[f"{traj_id.split('/')[0]}_first_sheaf_first_eigvec_covariance"] = wandb.Image(fig)
            plt.close(fig)

    run.log(img_dict)
            
            
         

# TODO fix the train val test split
# move validation code to it's own function
def train_residue_classifier(args_dict):
    # hyperparameters
    epsilon = 5.0 # in Angstroms
    learning_rate = 4e-3
    epochs = args_dict['epochs']
    val_ratio = 0.15
    test_ratio = 0.15
    num_nodes = 2000 # GRU's can't do more than a batch of 2**16
    hidden_dim = 64
    stalk_dim = 8
    num_blocks = 4
    masking_ratio = 1.0
    use_masking = masking_ratio < 1.0
    ablate_sheaves=args_dict["ablate_sheaves"]
    use_attention = not args_dict["ablate_attention"]
    num_heads = 4 if use_attention else 1
    run_name = args_dict["run_name"]
    restriction_map_type=args_dict["restriction_map_type"]
    seed=42
    num_timesteps = 128
    use_scheduler=False
    test_val = False
    use_profiler = False
    resume_model_name = args_dict["resume"]
    resume = args_dict["resume"] != ""
    nth_cross_val = args_dict["cross_val"]
    nth_cross_test = args_dict["cross_test"]

    assert nth_cross_val >= 0 and nth_cross_val < 5
    assert nth_cross_test >= 0 and nth_cross_test < 5

    ds_name = args_dict["ds_name"]
    other_model = args_dict["other_model"]
    
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 

    # login on HF
    login(token=os.environ["HF_TOKEN"])
    api = HfApi()

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    torch_rng = torch.Generator(); torch_rng = torch_rng.manual_seed(seed)
    np_rng = np.random.default_rng(seed=seed)

    # set up device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ATLAS
    if ds_name == "atlas":
        h5_path = os.path.abspath("atlas_data.h5")
        index = pd.read_csv(os.path.abspath("atlas_cross_val_index.csv"))

        val_mask = index["cross_val"] == nth_cross_val
        test_mask = index["cross_val"] == nth_cross_test

        val_pdbs = index[val_mask]["pdb"].to_list()
        test_pdbs = index[test_mask]["pdb"].to_list()
        train_pdbs = index[(~test_mask) & (~val_mask)]["pdb"].to_list()

    # mdCATH
    else:
        h5_path = os.path.abspath("mdcath_spinet_320_0.h5")
        index = pd.read_csv(os.path.abspath("mdcath_320_0_topology_split.csv"))

        train_pdbs = index[index["split"] == "train"]["domain"].tolist()
        test_pdbs = index[index["split"] == "test"]["domain"].tolist()
        val_pdbs = index[index["split"] == "validation"]["domain"].tolist()

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
            "num_nodes": num_nodes,
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
            "use_masking": use_masking,
            "other_model":other_model
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
    train_dataset = ResidueClassifierDataset(h5_path, np_rng, groups=train_pdbs,  epsilon=epsilon, fixed_length=num_timesteps, step=32, traj_len= 1000 if ds_name == "atlas" else None)
    val_dataset = ResidueClassifierDataset(h5_path, np_rng, groups=val_pdbs, epsilon=epsilon, fixed_length=num_timesteps, step=32, traj_len= 1000 if ds_name == "atlas" else None)
    test_dataset = ResidueClassifierDataset(h5_path, np_rng, groups=test_pdbs, epsilon=epsilon, fixed_length=num_timesteps, step=32, traj_len= 1000 if ds_name == "atlas" else None)

    # set up dataloaders
    train_loader = DataLoader(train_dataset, batch_sampler = ProteinSampler(train_dataset, num_nodes, rng=torch_rng), generator=torch_rng, num_workers=16, persistent_workers=True, worker_init_fn=ResidueClassifierDataset.worker_init_fn, collate_fn=lambda batch:train_dataset.graph_collate(batch), pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_sampler=ProteinSampler(val_dataset, num_nodes), num_workers=16, collate_fn=lambda batch:val_dataset.graph_collate(batch), pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_sampler=ProteinSampler(test_dataset, num_nodes), num_workers=16, collate_fn=lambda batch:val_dataset.graph_collate(batch), pin_memory=True)
    
    print(f"train expected groups: {len(train_dataset.groups)}")
    print(f"train actual groups: {len(set([train_dataset.index[idx][0] for batch in train_loader.batch_sampler.batches for idx in batch]))}")


    print(f"validation expected groups: {len(val_dataset.groups)}")
    print(f"validation actual groups: {len(set([val_dataset.index[idx][0] for batch in val_loader.batch_sampler.batches for idx in batch]))}")

    single_graph_val_loader = itertools.islice(DataLoader(val_dataset, shuffle=True, generator=torch_rng, batch_size=1, num_workers=16, collate_fn=lambda batch:val_dataset.graph_collate(batch), pin_memory=True, drop_last=False), 100)

    num_classes = len(ResidueClassifierDataset.AMINO_ACIDS)

    print([f"{key}, {item.shape}" for key, item in next(iter(train_loader)).to_data_list()[0].items() if hasattr(item,"shape")])

    # set up new diffusion model # TODO fix all this
    # ---------------------------------------------
    if other_model == "spinet":
        model = NodeSheafClassifier(
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
        #model = torch.jit.trace(model, Data(x = torch.rand(100, 128, 15, generator=torch_rng), edge_index = torch.randint(100, (2,200), generator=torch_rng), edge_attr=torch.randint(2, (200, 1), generator=torch_rng).to(float), y=torch.randint(20, (100,), generator=torch_rng), pos= torch.rand(100, 128, 4, 3, generator=torch_rng), frame_maps=torch.rand(100, 128, 3, 3, generator=torch_rng), node_mask= torch.zeros(100), lengths= torch.tensor([128])))

    elif other_model == "node_only": 
        model = NodeOnlyMLP(
             input_dim=15,
             hidden_dim=hidden_dim,
             num_classes=num_classes,
             dropout=0.2,
         ).to(DEVICE)
    else:
        model = RGNN(15, 13, hidden_dim, atoms=BACKBONE_ATOMS, frame_origin="CA").to(DEVICE)
    # Credit: Tomerikoo and Fabio Perez on StackOverflow
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run.log({"params": pytorch_total_params})

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

    # load_esmfold returns (tokenizer, model), in that order.
    # esmfold_model, esmfold_tokenizer = load_esmfold(device=DEVICE)
    esmfold_model = None
    esmfold_tokenizer = None

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
            for batch_idx, batch in enumerate(pbar):
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
            print(
                prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1)
            )  
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
        run_val(run, model, val_loader, val_dataset, epoch, DEVICE, crit, esmfold_tokenizer, esmfold_model, val_name="val", test_val=test_val)

    # Test Check
    run_val(run, model, test_loader, test_dataset, -1, DEVICE, crit, esmfold_tokenizer, esmfold_model, val_name="test", test_val=test_val)

    if not ablate_sheaves and other_model=="spinet":
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
    parser.add_argument('--resume', type=str, default="")
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--cross-val', type=int, default=0)
    parser.add_argument('--ds-name', type=str, default="atlas", choices=["atlas", "mdcath"])
    parser.add_argument('--other-model', type=str, default="spinet", choices=["rgnn", "spinet", "node_only"])
    parser.add_argument('--cross-test', type=int, default=4)

    args = parser.parse_args()
    train_residue_classifier(vars(args))

from torch_geometric.data import Dataset, download_url, Batch, Data
import torch
import os
from torch_geometric.loader import DataListLoader, DataLoader
from tqdm import tqdm
import random


class Cath(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, list_IDs, baseDIR, transform=None, pre_transform=None, pre_filter=None, pred_sasa=False,
                 max_length=None):
        super().__init__(baseDIR, transform, pre_transform, pre_filter)
        'Initialization'
        self.baseDIR = baseDIR
        self.pred_sasa = pred_sasa
        if max_length is None:
            self.list_IDs = list_IDs
        else:
            self.list_IDs = [
                ID for ID in tqdm(list_IDs, desc='filtering by max_length')
                if torch.load(self.baseDIR + ID).x.shape[0] <= max_length
            ]
            n_dropped = len(list_IDs) - len(self.list_IDs)
            if n_dropped > 0:
                print(f'Dropped {n_dropped}/{len(list_IDs)} proteins over max_length={max_length} residues')

    def len(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def get(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        data = torch.load(self.baseDIR + ID)
        del data['distances']
        del data['edge_dist']
        mu_r_norm = data.mu_r_norm
        extra_x_feature = torch.cat([data.x[:, 20:], mu_r_norm], dim=1)
        graph = Data(
            x=data.x[:, :20],
            extra_x=extra_x_feature,
            pos=data.pos,
            atom_pos=data.atom_pos,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            ss=data.ss[:data.x.shape[0], :],
            sasa=data.x[:, 20],
            # The protein id (``<pdb>_<chain>`` for ATLAS, the CATH domain for mdCATH), carried
            # through `Batch.from_data_list` as a per-graph list so trainer.py can label each
            # design in its `pred_seqs` W&B table -- matching what
            # scripts/train_residue_classifier.py logs from `Data.traj_id`.
            name=ID[:-3] if ID.endswith('.pt') else ID,
        )
        return graph

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import pandas as pd
from Bio.Data import IUPACData

class MotionClassifierDataset(Dataset):
    # ground truth order of amino acid indices. they must be capitalized
    AMINO_ACIDS = [code.upper() for code in IUPACData.protein_letters_3to1.keys()] + ["PYL", "SEC"] # add pyrrolysine and selenocysteine
    FEATURE_COLS = ["x","y","z", "dx", "dy", "dz"]
    # df should be loaded in with the pdb_id col added, and then validation set formed by splitting out along that column. Want to make a protein in the validation set has never been seen before
    def __init__(self, df):
        self.df = df
        self.groups = df.groupby(['timestep', "traj_id"])
        
    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        _, df = list(self.groups)[idx]
        return torch.from_numpy(df[self.__class__.FEATURE_COLS].to_numpy()), torch.tensor([self.__class__.AMINO_ACIDS.index(res) for res in df['residue'].to_list()], dtype=torch.long), torch.from_numpy(df["mask"].to_numpy())

    @staticmethod
    def pad_collate(batch):
        features, targets, masks = zip(*batch)
        features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)
        targets_padded = pad_sequence(targets, batch_first=True, padding_value=0l)
        masks_padded = pad_sequence(masks, batch_first=True, padding_value=False)

        return features_padded, targets_padded, masks_padded


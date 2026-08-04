import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import pandas as pd
from Bio.Data import IUPACData

class MotionClassifierDataset(Dataset):
    # ground truth order of amino acid indices. they must be capitalized
    AMINO_ACIDS = [code.upper() for code in IUPACData.protein_letters_3to1.keys()] + ["PYL", "SEC"] # add pyrrolysine and selenocysteine

    def __init__(self, conformations_df):
        self.df = conformations_df
        self.groups = conformations_df.groupby('motion_id')
        
    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        _, df = list(self.groups)[idx]
        return torch.from_numpy(df[["conf1_0", "conf1_1", "conf1_2"]].to_numpy()), torch.tensor(df['residue'].to_numpy(),dtype=torch.long)

    @staticmethod
    def pad_collate(batch):
        = zip(*batch)
        = pad_sequence(conformations1, batch_first=True, padding_value=0.0)

        return conformations1_padded



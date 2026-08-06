import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import pandas as pd
from Bio.Data import IUPACData
from torch_geometric.data import Data, Batch
from sheaf_utils import build_graph

class ResidueClassifierDataset(Dataset):
    # ground truth order of amino acid indices. they must be capitalized
    AMINO_ACIDS = [code.upper() for code in IUPACData.protein_letters_3to1.keys()] + ["PYL", "SEC"] # add pyrrolysine and selenocysteine
    FEATURE_COLS = ["dx", "dy", "dz", "bond_ang", "bond_len"]
    POS_COLS = ["x","y","z"]
    # df should be loaded in with the pdb_id col added, and then validation set formed by splitting out along that column. Want to make a protein in the validation set has never been seen before
    # TODO plot histogram of epsilon
    def __init__(self, df, epsilon=5.0):
        self.df = df
        self.groups = df.groupby(['timestep', "traj_id"])
        self.epsilon = epsilon 

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        _, df = list(self.groups)[idx]
        return build_graph(torch.from_numpy(df[self.__class__.POS_COLS].to_numpy()), torch.from_numpy(df[self.__class__.FEATURE_COLS].to_numpy()), torch.tensor([self.__class__.AMINO_ACIDS.index(res) for res in df['residue'].to_list()], dtype=torch.long), torch.from_numpy(df["mask"].to_numpy()), self.epsilon)
 
 
    @staticmethod
    def graph_collate(batch):
        return Batch.from_data_list(batch)

        

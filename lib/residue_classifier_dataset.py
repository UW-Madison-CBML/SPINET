import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import pandas as pd
from Bio.Data import IUPACData

class MotionClassifierDataset(Dataset):
    # this will be the ground truth order of pocket motion classes
    # PE = Pocket-expanding
    # PS = Pocket-shrinking
    # PF = Pocket-fusing
    # PC = Pocket-creating
    # OM = Other motion
    MOTION_CLASSES = ["PE","PS","PF","PC","OM"]
    # ground truth order of amino acid indices. they must be capitalized
    AMINO_ACIDS = [code.upper() for code in IUPACData.protein_letters_3to1.keys()] + ["PYL", "SEC"] # add pyrrolysine and selenocysteine
    def __init__(self, conformations_df):
        # load each protein and get it's 3d structure sequence and amino acid sequence 
        # self.df:
        # motion_id, residue, conf1_0 .. conf2_2, ligand, motion_class, etc.
        # motion_id = pdb_1 + pdb_2 these will serve as UUID's for protein motions
        self.df = conformations_df
        self.groups = conformations_df.groupby('motion_id')
        

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        _, df = list(self.groups)[idx]
        return torch.from_numpy(df[["conf1_0", "conf1_1", "conf1_2"]].to_numpy()), torch.from_numpy(df[["conf2_0", "conf2_1", "conf2_2"]].to_numpy()), torch.tensor(df['residue'].to_numpy(),dtype=torch.long), torch.tensor(self.__class__.MOTION_CLASSES.index(df.iloc[0]["motion_class"]), dtype=torch.long)
    @staticmethod
    def pad_collate(batch):
        conformations1, conformations2, residues, motion_classes = zip(*batch)
        conformations1_padded = pad_sequence(conformations1, batch_first=True, padding_value=0.0)
        conformations2_padded = pad_sequence(conformations2, batch_first=True, padding_value=0.0)
        # need to pad with 0 otherwise F.one_hot() will fail in the training loop 
        # this will be packed so should not affect gradients
        residues_padded = pad_sequence(residues, batch_first=True, padding_value=0)

        # lengths will be the same for both lists of conformations
        lengths = torch.tensor([len(conf) for conf in conformations1])
        
        motion_classes = torch.stack(motion_classes, dim = 0)

        return conformations1_padded, conformations2_padded, residues_padded, motion_classes, lengths



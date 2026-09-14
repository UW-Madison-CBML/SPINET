from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB import PDBIO
import requests
import pandas as pd
from pdb_api import retrieve_pdb_file
if __name__ == "__main__":

    pdbs_df = pd.read_csv("atlas_cross_val_index.csv")
    pdbs = pdbs_df["pdb"].str.slice(0, 4).to_list()
    cif_paths = []
    os.makedirs("cifs", exist_ok=True)
    for pdb in pdbs:
        cif_paths.append(retrieve_pdb_file(pdb), parent_dir="cifs")
    
        
        parser = MMCIFParser()
        io = PDBIO()

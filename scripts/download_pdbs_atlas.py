from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBIO import PDBIO, Select
import requests
import pandas as pd
from pdb_api import retrieve_pdb_file
from tqdm import tqdm

class SelectChain(Select):
    def __init__(self, chain):
        super().__init__()
        self.chain = chain
    def accept_chain(self, chain):
        return chain == self.chain
if __name__ == "__main__":

    pdbs_df = pd.read_csv("atlas_cross_val_index.csv")
    pdbs = pdbs_df["pdb"].str.slice(0, 4).to_list()
    chains = pdbs_df["pdb"].str.slice(5, 6).to_list()

    os.makedirs("cifs", exist_ok=True)
    os.makedirs("atlas_pdbs", exist_ok=True)
    
    parser = MMCIFParser()

    io = PDBIO()

    for pdb, chain in tqdm(zip(pdbs, chains)):
        cif_path = retrieve_pdb_file(pdb, parent_dir="cifs")
         
        structure = parser.get_structure(f"{pdb}_{chain}", cif_path)
        select = SelectChain(chain)

        io.set_structure(structure)
        io.save(os.path.join("atlas_pdbs", f"{pdb}_{chain}"), select=select)


        
                

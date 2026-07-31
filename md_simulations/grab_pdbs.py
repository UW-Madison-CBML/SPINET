import os
import sys
import urllib.request
from Bio.PDB import PDBParser, MMCIFParser, PDBIO, Select
# TODO figure out case sensitivity here
class ChainSelectOnlyProtein(Select):
    def __init__(self, chain_id):
        self.chain_id = chain_id
    def accept_chain(self, chain):
        return chain.id == self.chain_id
    def accept_residue(self, residue):
        return residue.id[0] == ' '

def process_protein(pdb_id, chain_id=None):
    pdb_id = pdb_id.lower()
    pdb_file = f"{pdb_id}.pdb"
    cif_file = f"{pdb_id}.cif"
    
    try:
        print(f"Trying standard PDB download for {pdb_id.upper()}...")
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb", pdb_file)
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, pdb_file)
        if(len(structure.get_atoms()) == 0):
            raise ValueError("no atoms reverting to cif")
    except Exception:
        print(f"PDB unavailable or too large. Fetching mmCIF format...")
        if os.path.exists(pdb_file): os.remove(pdb_file) 
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb_id.upper()}.cif", cif_file)
        parser = MMCIFParser()
        structure = parser.get_structure(pdb_id, cif_file)

    io = PDBIO()
    io.set_structure(structure)

    if chain_id:
        output_file = os.path.join("inputs",f"{pdb_id.upper()}_{chain_id.upper()}.pdb")
        io.save(output_file, select=ChainSelectOnlyProtein(chain_id.upper()))
        print(f"Successfully extracted chain {chain_id}: {output_file}\n")
    else:
        chain_id = structure.get_chains()[0]
        output_file = os.path.join("inputs",f"{pdb_id.upper()}.pdb")
        io.save(output_file, select=ChainSelectOnlyProtein(chain_id.upper())) # implicit chain? TODO
        print(f"Successfully extracted all chains into single file: {output_file}\n")
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, output_file)

    n_atoms = sum(1 for atom in structure.get_atoms())
    n_residues = sum(1 for res in structure.get_residues())
    n_chains = sum(1 for chain in structure.get_chains())

    print(f"Atoms: {n_atoms}, Residues: {n_residues}, Chains: {n_chains}")
        

"""
        except Exception as e:
            print(f"Structure exceeds standard PDB file limits. Splitting chains automatically...")
            if os.path.exists(output_file): os.remove(output_file)
            
            for model in structure:
                for chain in model:
                    chain_output = f"{pdb_id}_{chain.id}.pdb"
                    io.save(chain_output, select=ChainSelect(chain.id))
                    print(f" -> Created: {chain_output}")
            print()
"""
if __name__ == "__main__":
    for arg in sys.argv[1:]:
        if "_" in arg:
            pdb_id, chain_id = arg.split("_")
            process_protein(pdb_id, chain_id=chain_id)
        else:
            process_protein(arg)

    

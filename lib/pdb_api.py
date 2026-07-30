import requests 
import pandas as pd
import numpy as np
import json
from Bio import PDB, Align
from Bio.SeqUtils import seq1, seq3
import sys
import time
import os
#TODO: so we need to load in the amino acid sequence from uniprot, and get the two structures from pdb. Then we align them    
    
def retrieve_pdb_file(pdb_id, file_format = "cif", parent_dir="./"):
    url = f"https://files.rcsb.org/download/{pdb_id.lower()}.{file_format}"
    
    # TODO remove in case of anonymization
    headers = {
        "User-Agent": "jlundsgaard@wisc.edu"
    }
    file_path = os.path.abspath(os.path.join(parent_dir, f"{pdb_id}.{file_format}"))

    i = 0
    attempts = 8
    delay = 1
    while(i < attempts):
        try:
            if (response := requests.get(url, headers=headers)).status_code != 200:
                time.sleep(delay)
            else:
                with open(file_path, "w") as file:
                    file.write(response.text)
                return file_path
        except requests.exceptions.ReadTimeout:
            time.sleep(delay)
        i += 1
        delay *= 1.3
    raise ValueError(f"{pdb_id} could not be accessed at {url}: error code {response.status_code}")


def load_pdb(pdb_plus_chain):
    # pdb ids are sometimes formatted like this
    pdb_plus_chain = pdb_plus_chain.replace(":", "_")
    if("_" not in pdb_plus_chain):
        print(f"no chain id: {pdb_plus_chain}")
        pdb_id = pdb_plus_chain
        chain_id = ""
    else:
        pdb_id, chain_id = pdb_plus_chain.split("_")
    pdb_id = pdb_id[:4].upper()

    # uncomment if using biopython PDB for api access
    #pdbl = PDB.PDBList()
    #file_path = pdbl.retrieve_pdb_file(pdb_id)

    file_path = retrieve_pdb_file(pdb_id)

    # uncomment if using old .pdb format
    # parser = PDB.PDBParser(QUIET=True)
    parser = PDB.MMCIFParser(QUIET=True) 
    structure = parser.get_structure(pdb_id, file_path) 
    first_structure = structure[0]
    all_chains = list(first_structure.get_chains())

    if chain_id and chain_id in all_chains:
        chain = first_structure[chain_id]  
    else:
        if all_chains:
            chain_id = all_chains[0].id.strip()
        else:
            chain_id = "A" # otherwise fall back to A, this does not seem to happen much
        chain_id = chain_id.upper()
        chain = first_structure[chain_id]
    atoms = {}
    res_names = {}
    for residue in chain:
        if residue.id[0] != " ":
            continue
        
        res_num = residue.id[1]
        
        if "CA" in residue:
            res_names[res_num] = residue.get_resname()

            atoms[res_num] = residue["CA"]
            
    return atoms, res_names

def pdb_chain_to_uniprot_id(pdb_plus_chain):

    pdb_plus_chain = pdb_plus_chain.replace(":", "_")

    if "_" not in pdb_plus_chain:
        # Fall back to chain 'A'
        pdb_id = pdb_plus_chain
        target_chain = "A"
    else:
        pdb_id, target_chain = pdb_plus_chain.split("_")
    
    pdb_id = pdb_id.lower()
    target_chain = target_chain.upper()

    # Query SIFTS API
    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id}"
    response = requests.get(url)


    if response.status_code == 200:
        data = response.json()
        try:
            uniprot_dict = data[pdb_id]['UniProt']
        
            for uniprot_id, mapping_data in uniprot_dict.items():
                for mapping in mapping_data['mappings']:
                    if mapping['chain_id'] == target_chain:
                        return uniprot_id # found exact match
            
            print(f"Chain {target_chain} not found in UniProt mappings for PDB {pdb_id.upper()}.")
            return None
        
        except KeyError:
            print(f"Mapping structure missing for PDB {pdb_id.upper()}.")
            return None

    else:
        print(f"Failed to fetch data for {pdb_id.upper()}. HTTPS Status: {response.status_code}")
        return None  

def get_uniprot_sequence(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    response = requests.get(url)

    if response.status_code == 200:
        # FASTA format
        fasta_lines = response.text.strip().split('\n')
        sequence = ''.join(fasta_lines[1:])
        return sequence
    else:
        print(f"Error fetching data: HTTP {response.status_code}")
        return None
    
def map_pdb_to_uniprot(atoms, res_names, uniprot_seq:str):
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -1

    # PDB sequence
    seq_list = [(num, seq1(name)) for num, name in res_names.items() if seq1(name) != 'X']

    seq_str = "".join([char for _, char in seq_list])

    # uniprot seq is already string
    alignment = aligner.align(uniprot_seq, seq_str)[0]
    uniprot_blocks, pdb_blocks = alignment.aligned[:2]

    aligned_atoms = [None] * len(uniprot_seq)

    for u_block, p_block in zip(uniprot_blocks, pdb_blocks):
        for i in range(u_block[1] - u_block[0]):
            u_idx = u_block[0] + i
            p_idx = p_block[0] + i

            res_num = seq_list[p_idx][0]
            if res_num in atoms:
                aligned_atoms[u_idx] = atoms[res_num]
    
    return aligned_atoms



def load_motion_structures(pdb1, pdb2, uniprot_seq:str):

    # uniprot_seq: single letter residues 
    atoms1, res_names1 = load_pdb(pdb1)
    atoms2, res_names2 = load_pdb(pdb2)

    aligned_atoms1 = map_pdb_to_uniprot(atoms1, res_names1, uniprot_seq)
    aligned_atoms2 = map_pdb_to_uniprot(atoms2, res_names2, uniprot_seq)

    intersecting_atoms1 = []
    intersecting_atoms2 = []
    intersection_residues = []

    for a1, a2, res in zip(aligned_atoms1, aligned_atoms2, uniprot_seq):
        if a1 is not None and a2 is not None:
            intersecting_atoms1.append(a1)
            intersecting_atoms2.append(a2)
            intersection_residues.append(seq3(res))
    
    super_imposer = PDB.Superimposer()
    super_imposer.set_atoms(intersecting_atoms1, intersecting_atoms2)

    super_imposer.apply(intersecting_atoms2) 

    conformation1 = np.array([atom.get_coord() for atom in intersecting_atoms1])
    conformation2 = np.array([atom.get_coord() for atom in intersecting_atoms2])

        
    return conformation1, conformation2, intersection_residues

def load_motion_structures_no_uniprot(pdb1, pdb2):
    atoms1, res_names1 = load_pdb(pdb1)
    atoms2, res_names2 = load_pdb(pdb2)

    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -1

    seq1_list = [(num, seq1(name)) for num, name in res_names1.items() if seq1(name) != 'X']
    seq2_list = [(num, seq1(name)) for num, name in res_names2.items() if seq1(name) != 'X']
    
    seq1_str = "".join([char for _, char in seq1_list])
    seq2_str = "".join([char for _, char in seq2_list])

    intersecting_atoms1 = []
    intersecting_atoms2 = []

    alignment = aligner.align(seq1_str, seq2_str)[0]  
    blocks1, blocks2 = alignment.aligned[:2]


    for block1, block2 in zip(blocks1, blocks2):
        for idx in range(block1[1] - block1[0]): # block1[1] - block1[0] = block2[1] - block2[0]
            idx1 = idx + block1[0]
            idx2 = idx + block2[0]

            atom1 = atoms1[seq1_list[idx1][0]]
            atom2 = atoms2[seq2_list[idx2][0]]
            if(atom1.get_parent().get_resname() == atom2.get_parent().get_resname()):
                intersecting_atoms1.append(atom1)
                intersecting_atoms2.append(atom2)
            
    
    super_imposer = PDB.Superimposer()
    super_imposer.set_atoms(intersecting_atoms1, intersecting_atoms2)

    super_imposer.apply(intersecting_atoms2) 

    
    return intersecting_atoms1, intersecting_atoms2, [atom.get_parent().get_resname() for atom in intersecting_atoms1]
# for temporary use
if __name__ == "__main__":
    pdb_list = [
    '4cfo',
    '4cfp',
    '6d6f',
    '4tws',
    '5a1r',
    '4ny4',
    '3kjr',
    '3k2h',
    '1fit',
    '6fit',
    '2bjs',
    '1odn',
    '2qyo',
    '4h4t',
    '4h50',
    '5tcg',
    '5tci',
    '4k4r',
    '4k4p',
    '5lcq',
    '5n1n',
    '3v4n',
    '1ysl',
    '4rzu',
    '4rzu']

    for i in pdb_list:
        uniprot_id = pdb_chain_to_uniprot_id(i)
        uniprot_seq = get_uniprot_sequence(uniprot_id)
        print(load_motion_structures(i, i, uniprot_seq))
    

import pandas as pd
import sys
# for now if  on chtc comment this
sys.path.append("../lib")
from pdb_api import load_motion_structures, get_uniprot_sequence, load_motion_structures_no_uniprot
from motion_classifier_dataset import MotionClassifierDataset
import os
from tqdm import tqdm

# Import biophysical properties and mappings directly from Biopython
from Bio.Data.IUPACData import protein_letters_3to1, protein_weights
from Bio.SeqUtils.ProtParamData import kd as hydropathy_index
from Bio.SeqUtils.ProtParamData import Flex as flexibility_index

PHYSIOLOGICAL_CHARGE = {
    'R': 1.0,  # Arginine
    'K': 1.0,  # Lysine
    'H': 0.1,  # Histidine
    'D': -1.0, # Aspartic Acid
    'E': -1.0  # Glutamic Acid
}

def main(use_uniprot):
    # the last two columns seem to be empty
    columns = ['uniprot_ID', 'pdb_1', 'pocket_size_free', 'pdb_2', 'ligand', 'pocket_size_bound', 'motion_class', 'motion_residues', 'RMSD_pocket']
    free_bound_df = pd.read_csv(os.path.abspath("free_bound_pocket.csv"),header=0)
    dif_ligand_df = pd.read_csv(os.path.abspath("bound_dif_ligand_pocket.csv"), header=0)

    # drop the last cols, free-bound has extra col
    free_bound_df = free_bound_df.iloc[:,:-2]
    dif_ligand_df = dif_ligand_df.iloc[:,:-1]

    # rename the columns
    free_bound_df.columns = columns
    dif_ligand_df.columns = columns

    df = pd.concat([free_bound_df, dif_ligand_df], axis=0, ignore_index=True)
    df = df[["pdb_1", "pdb_2", "motion_class", "uniprot_ID"]] # remove unecessary rows before we dropna

    df = df.dropna()
    # universal motion identifier
    df["motion_id"] = df['pdb_1'] + "-" + df['pdb_2'] # is this actually a primary key?

    groups = []
    pbar = tqdm(list(df.iterrows()))

    for idx, row in pbar:
        pbar.set_postfix(
                    motion1=f"{row['pdb_1']}",
                    motion2=f"{row['pdb_2']}",
                )
        try:
            if(use_uniprot):
                uniprot_id = row['uniprot_ID']
                if not uniprot_id:
                    print(f"Skipping {row['motion_id']}: Could not map to UniProt.")
                    continue

                uniprot_seq = get_uniprot_sequence(uniprot_id)
                if not uniprot_seq:
                    print(f"Skipping {row['motion_id']}: Could not fetch FASTA.")
                    continue
            
            # padded, uniprot-aligned coordinates
            # save aligned but ground truth pdb files
            if(use_uniprot):
                conformation1, conformation2, residues = load_motion_structures(row["pdb_1"], row["pdb_2"], uniprot_seq, "pdbs")
            else:
                conformation1, conformation2, residues = load_motion_structures_no_uniprot(row["pdb_1"], row["pdb_2"], "pdbs")

            residue_indices = []
            for res in residues:
                try:
                    residue_indices.append(MotionClassifierDataset.AMINO_ACIDS.index(res.upper()))
                except ValueError:
                    residue_indices.append(-1)
            
            # ---> Biopython Feature Extraction <---
            hydropathy_vals = []
            weight_vals = []
            flexibility_vals = []
            charge_vals = []

            for res in residues:
                # Biopython's 3-to-1 map uses capitalized 3-letter codes (e.g., 'Ala', 'Arg')
                res_3 = res.capitalize() 
                # Convert to 1-letter code; use 'X' for unknown/non-standard residues
                res_1 = protein_letters_3to1.get(res_3, 'X')
                
                # Fetch properties using the 1-letter code (defaulting to neutral values if not found)
                hydropathy_vals.append(hydropathy_index.get(res_1, 0.0))
                weight_vals.append(protein_weights.get(res_1, 0.0))
                flexibility_vals.append(flexibility_index.get(res_1, 1.0))
                charge_vals.append(PHYSIOLOGICAL_CHARGE.get(res_1, 0.0))
            
            res_df = pd.DataFrame({
                "residue": residue_indices,
                "motion_class": row["motion_class"],
                "motion_id": row["motion_id"],
                "res_name": list(residues),
                "hydropathy": hydropathy_vals,
                "weight": weight_vals,
                "flexibility": flexibility_vals,
                "charge": charge_vals
            })

            # construct x,y,x coords
            conformation1_df = pd.DataFrame(conformation1, columns=["conf1_0", "conf1_1", "conf1_2"], index=res_df.index)
            conformation2_df = pd.DataFrame(conformation2, columns=["conf2_0", "conf2_1", "conf2_2"], index=res_df.index)

            conformation_df = pd.concat([res_df, conformation1_df, conformation2_df], axis=1)
            groups.append(conformation_df)
            
        except (ValueError, KeyError, AttributeError,ZeroDivisionError) as e:
            print(f"error skipping row {row['motion_id']}: {e}")

    print("motions: ", len(groups))
    df = pd.concat(groups, axis=0, ignore_index=True)
    print("rows: ", len(df))
    df.to_csv("motions.csv")
    
if __name__ == "__main__":
    import sys
    main(bool(sys.argv[1]))

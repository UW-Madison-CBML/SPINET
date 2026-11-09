#!/bin/bash
#get_kabsch_rmsd.sh <seq_csv stem> <trajectory store file name, or 'none'>
set -e

# mdCATH references are the CATH domain files (lib/relaxed_pdb.load_cath_structures).
if [[ -f mdcath_pdbs.tar.gz ]]; then
    mkdir -p pdbs && tar -xf mdcath_pdbs.tar.gz -C pdbs
fi

H5_ARGS=()
if [[ -n "$2" && "$2" != "none" ]]; then
    H5_ARGS=(--traj-h5 "$2")
fi

python get_kabsch_rmsd.py "$1" "${H5_ARGS[@]}"

tar -czf "$1"_pred_pdbs.tar.gz pred_pdbs/

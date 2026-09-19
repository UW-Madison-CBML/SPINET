#!/bin/bash
#get_kabsch_rmsd.sh

python get_kabsch_rmsd.py "$1"

tar -xvf "$1"_pred_pdbs.tar.gz pred_pdbs/

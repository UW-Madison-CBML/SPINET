#!/bin/bash
#download_pdbs.sh

mkdir pdbs/

python download_pdbs.py --ids "$1".txt

tar -czvf "$1".tar.gz pdbs/


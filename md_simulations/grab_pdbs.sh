#!/bin/bash
mkdir inputs
pip install biopython --quiet
python grab_pdbs.py "$@"
ls -R
tar -czvf inputs.tar.gz inputs/

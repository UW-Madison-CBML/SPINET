#!/bin/bash
mkdir pdbs
python build_motion_class_index.py
tar -czvf pdbs.tar.gz pdbs/
rm *.pdb
rm *.cif

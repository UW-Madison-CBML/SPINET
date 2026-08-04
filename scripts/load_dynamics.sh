#!/bin/bash

mkdir md_data
pip install pandas requests

python load_dynamics.py

tar -czvf md_data.tar.gz md_data/

rm -rf md_data/

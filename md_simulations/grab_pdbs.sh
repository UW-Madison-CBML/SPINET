#!/bin/bash
mkdir inputs
for var in "$@"
do
    curl -o inputs/"$var".pdb https://rcsb.org
done
tar -czvf inputs.tar.gz inputs/

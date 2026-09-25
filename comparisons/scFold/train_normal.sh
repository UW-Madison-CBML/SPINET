#!/bin/bash

tar -xvf ScFold.tar.gz
mkdir ScFold/data/cath

mv chain_set_0.jsonl ScFold/data/cath/chain_set.jsonl
mv chain_set_splits_0.json ScFold/data/cath/chain_set_splits.json

cd ScFold/


python3 main.py "$@"


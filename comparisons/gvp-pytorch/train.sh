#!/bin/bash

tar -xvf gvp-pytorch.tar.gz
cd gvp-pytorch/
mv ../train.py ./
tar -xvf ../surffold_data.tar.gz


python train.py

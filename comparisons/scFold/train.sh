#!/bin/bash

tar -xvf ScFold.tar.gz

cd ScFold/

python3 main.py --epoch 8

tar -czvf ../plots.tar.gz plots/

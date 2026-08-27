#!/bin/bash/

tar -xvf md_data.tar.gz

mkdir surffold_data/

cd surffold_data/

mkdir train/ validation/ test/

cd ..

python split_dynamics.py

tar -czvf surffold_data.tar.gz surffold_data/

rm -rf md_data/ surffold_data/

#!/bin/bash
rm ScFold.tar.gz
git clone https://github.com/JensLundsgaard/ScFold.git

cd ScFold/
git checkout main

cd ..
#git clone https://github.com/jczhongcs/ScFold.git

tar -czvf ScFold.tar.gz ScFold/
rm -rf ScFold/

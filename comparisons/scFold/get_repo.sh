#!/bin/bash
rm ScFold.tar.gz
git clone https://github.com/JensLundsgaard/ScFold.git
tar -czvf ScFold.tar.gz ScFold/
rm -rf ScFold/

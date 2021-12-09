#!/bin/bash

set -ue
pip install gdown

mkdir data
cd data
gdown https://drive.google.com/uc?id=1MGnaAfbckUmigGUvihz7uiHGC6rBIbvr
tar -xf dataset.tar.gz

rm dataset.tar.gz
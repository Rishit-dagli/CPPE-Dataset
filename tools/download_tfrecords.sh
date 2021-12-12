#!/bin/bash

set -ue
pip install gdown

mkdir tfrecords
cd tfrecords
gdown https://drive.google.com/uc?id=1jBHxybNWx4uhLxWxyguHpmjzz73YMQdG
tar -xf tfrecords.tar.gz

rm tfrecords.tar.gz
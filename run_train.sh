#!/bin/bash

# set gpu id to use
export CUDA_VISIBLE_DEVICES=0

# set python path according to your actual environment
pythonpath='python3'

# set parameters
data_dir=./data/CamRest
save_dir=./models/camrest
num_epochs=10
pre_epochs=10

${pythonpath} main.py --data_dir=${data_dir} --save_dir=${save_dir} --num_epochs=${num_epochs} --pre_epochs=${pre_epochs}

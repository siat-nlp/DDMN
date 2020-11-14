#!/bin/bash

# set python path
pythonpath='python3 -m'

# set dir
data_name='camrest'     # ['kvr', 'camrest', 'multiwoz']
data_dir=./data/CamRest
eval_dir=./outputs/camrest/best.model

${pythonpath} tools.eval --data_name=${data_name} --data_dir=${data_dir} --eval_dir=${eval_dir}

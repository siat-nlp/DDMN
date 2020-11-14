# DDMN
This repository contains data and code for the COLING'2020 paper "[Dual Dynamic Memory Network for End-to-End Multi-turn Task-oriented Dialog Systems]()".

In this paper, we propose a Dual Dynamic Memory Network (DDMN) for multi-turn dialog generation, which maintains two core components: dialog memory manager and KB memory manager. The dialog memory manager dynamically expands the dialog memory turn by turn and keeps track of dialog history with an updating mechanism, which encourages the model to filter irrelevant dialog history and memorize important newly coming information. The KB memory manager shares the structural KB triples throughout the whole conversation, and dynamically extracts KB information with a memory pointer at each turn.

## Requirements
The implementation is based on Python 3.x. To install the dependencies used in this project, please run:
```
pip install -r requirements.txt
```

## Quickstart

### Step 1: Training
For different datasets, please first set up the following parameters in the script `run_train.sh`:
```
data_dir=[xxx]        # directory of the specific dataset
save_dir=[xxx]        # directory to store trained models
```
and then run:
```
sh run_train.sh
```
Note that more arguments for training can be found in the `main.py`. 

For self-critical sequence training, please set up `num_epochs` larger than `pre_epochs` but no larger than 3 epochs, since self-critical sequence training for a long time might be unstable sometimes.

### Step 2: Testing
For different datasets, please first set up the following parameters in the script `run_test.sh`:
```
data_dir=[xxx]        # directory of the specific dataset
save_dir=[xxx]        # directory of the rrtrained models
output_dir=[xxx]      # directory to store generation results
```
and then run:
```
sh run_test.sh
```
Note that more arguments for testing can be found in the `main.py`. 

### Step 3: Evaluation
For different datasets, please first set up the following parameters in the script `run_eval.sh`:
```
data_name=[xxx]      # ['kvr', 'camrest', 'multiwoz']
data_dir=[xxx]       # directory of the specific dataset
eval_dir=[xxx]       # directory of the generation output
```
and then run:
```
sh run_eval.sh
```
#!/bin/sh
module load anaconda3/2020.11
conda activate /ocean/projects/tra220029p/tjin1/anaconda3/envs/e2e-ser
python train.py --conf conf/batch_disable_all_input.yaml --tag debug_all_input_batch_disable
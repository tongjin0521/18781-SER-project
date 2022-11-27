#!/bin/sh
module load anaconda3/2020.11
conda activate /ocean/projects/tra220029p/tjin1/anaconda3/envs/e2e-asr
python train.py --conf conf/batch_with_hf.yaml --tag handcrafted_features_debug
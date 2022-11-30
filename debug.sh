#!/bin/sh
module load anaconda3/2020.11
conda activate /ocean/projects/tra220029p/tjin1/anaconda3/envs/e2e-ser
python train.py --conf conf/batch_disable_only_hf.yaml --tag only_handcrafted_features_debug
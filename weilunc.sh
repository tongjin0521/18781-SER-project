#!/bin/sh
module load anaconda3/2020.11
conda activate /ocean/projects/tra220029p/tjin1/anaconda3/envs/e2e-ser
python train.py --conf conf/bmcc_0.yaml --tag train_bmcc_0.0_lr1e-3_ep60_mcc

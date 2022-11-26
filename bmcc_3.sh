#!/bin/sh
module load anaconda3/2020.11
conda activate /ocean/projects/tra220029p/tjin1/anaconda3/envs/e2e-asr
python train_batch.py --conf conf/bmcc_3.yaml --tag bmcc_3_lr1e-3_ep60_mcc

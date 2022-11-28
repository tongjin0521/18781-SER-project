#!/bin/sh
module load anaconda3/2020.11
conda activate /ocean/projects/tra220029p/tjin1/anaconda3/envs/e2e-ser
python train_finetune.py --conf conf/s3prl_finetune.yaml --tag godhelpme

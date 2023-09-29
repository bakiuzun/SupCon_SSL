#!/bin/bash -l
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --time 47:00:00
#SBATCH --mem 20G
#SBATCH --constraint a6000
#SBAtCH -w sn4
#SBATCH --output TEST.out
setcuda 11.0
conda activate myenv

python -u dfc_finetune.py \
    --model_path "/share/projects/ottopia/ssl_Baki/poid/dfc_100SUP_100_CROSS_1_IMG224_BATCH32__average55.64_overall72.60_epoch175.pth" \
    --epochs 201 \
    --batch_size 32  \
    --use_sup_con  \
    --delta_sup 1.0 \
    --delta_lin 1.0 \
    

#python -u dfc_dataset.py
#python -u new_dataset.py
#python -u  resnet_eval.py



#!/bin/bash -l
#SBATCH --gres=gpu:1
#SBATCH --constraint titan
#SBATCH --cpus-per-gpu 8
#SBATCH --mem 20G
#SBATCH --time 24:00:00
#SBATCH --output DENEME4.out


## default setting to test the downloaded model from github , METER ML dataset 
## MODEL_PATH="the path where you have saved the model"
## EVAL_MODE=linear
##SCENARIO=s1s2_fused
## python -u meterml_finetune.py \
## $MODEL_PATH  \
##  --eval_mode $EVAL_MODE \
##  --scenario "$SCENARIO" \
##  --dont_include_negatives \
##  --model dual \
##  --n_epochs 121 \
## --seed 10 \
##  --sup_delta 1. \
##  --cross_delta 1. \
##  --data_ratio 1. \
##  --batch_size 32 \



setcuda 11.0
conda activate myenv

base="/share/projects/ottopia/ssl_Baki/poid/"
model="meter-ml_directory_model_dual_120ep_no_pairs_0.pckl"
model="meterml_100SUP_100CROSS_4run__average_61.64_epoch_107.pth"
MODEL_PATH="${base}${model}"

EVAL_MODE=linear
SCENARIO=s1s2_fused
python -u meterml_finetune.py \
  $MODEL_PATH  \
  --eval_mode $EVAL_MODE \
  --scenario "$SCENARIO" \
  --dont_include_negatives \
  --model dual \
  --n_epochs 121 \
  --seed 10 \
  --save_path  "/share/projects/ottopia/ssl_Baki/poid/meterml_100SUP_100CROSS_6run_"  \
  --sup_delta 1. \
  --cross_delta 1. \
  --data_ratio 1. \
  --batch_size 32 \


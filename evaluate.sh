#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 5
#SBATCH --chdir=/share/home/e2100317/ottopia/meter-ml
#SBATCH --constraint 2080
#SBATCH --cpus-per-gpu 8
#SBATCH --mem 24G


set -o errexit
set -o pipefail
set -o nounset

MODEL=$1
EVAL_MODE=${2-finetune}
SCENARIO=${3-s1s2}

python -u evaluate.py \
  random \
  --eval_mode $EVAL_MODE \
  --dont_include_negatives \
  --scenario "$SCENARIO" \
  --n_epochs 100 \
  | tee evaluate_$(basename -s .pckl model)_${SCENARIO}_${EVAL_MODE}.out

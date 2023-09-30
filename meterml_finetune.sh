

## default setting to test the downloaded model from github , METER ML dataset 
## MODEL_PATH="the path where you have saved the model"
EVAL_MODE=finetune
SCENARIO=s1s2_fused
python -u meterml_finetune.py \
  $MODEL_PATH  \
  --eval_mode $EVAL_MODE \
  --scenario "$SCENARIO" \
  --dont_include_negatives \
  --model dual \
  --n_epochs 121 \
 --seed 10 \
 --use_sup_con\
  --sup_delta 1. \
  --cross_delta 1. \
  --data_ratio 1. \
  --batch_size 32 \

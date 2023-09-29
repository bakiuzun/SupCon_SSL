---
To get the dataset run `download_all.sh`.
then `shared_encoder.py` to pretrain and `evaluate.py` to evaluate the
pretrained encoder.

MODEL_PATH="model_dual_200ep_pairs_0.pckl" # replace with the checkpoint
python evaluate.py $MODEL_PATH --scenario s1naip_fused # finetune + evaluate on s1naip_fused

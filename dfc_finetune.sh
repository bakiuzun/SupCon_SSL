

python -u dfc_finetune.py \
    --model_path "mymodelpath.pth" \
    --epochs 201 \
    --batch_size 32  \
    --use_sup_con  \
    --delta_sup 1.0 \
    --delta_lin 1.0 \
    



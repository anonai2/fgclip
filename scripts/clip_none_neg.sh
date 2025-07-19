export PYTHONPATH=/fgclip/src
export CUDA_VISIBLE_DEVICES=1

python ./training/main_ckpt_finetune.py \
    --model ViT-B-16 \
    --pretrained laion400m_e32 \
    --train-data '/fgclip/synthtiger/synthtiger_v1.0_data/synthtiger_v1.0/gt_hn_4.json' \
    --train-mode "None_neg" \
    --dataset-type clip_json \
    --batch-size 128 \
    --precision amp \
    --workers 8 \
    --epochs 3 \
    --cmr_loss_weight 0.2 \
    --imc_loss_weight 0.3 \
    --hndc_loss_weight 1.0 \
    

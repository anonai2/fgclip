export PYTHONPATH=/home/work/hard_negative_ocr/code/neg_clip/src
export CUDA_VISIBLE_DEVICES=0,1
# python -m training.main_ckpt_finetune_mlp \
torchrun --nproc_per_node=2 ./training/main_ckpt_finetune.py \
    --model ViT-L-14 \
    --pretrained openai \
    --train-data '/home/work/cc_ocr/synthtiger/synthtiger_v1.0_data/synthtiger_v1.0/gt_hn_4.json' \
    --train-mode hn_far \
    --dataset-type clip_json_da \
    --batch-size 128 \
    --precision amp \
    --workers 8 \
    --epochs 3 \
    --cmr_loss_weight 0.2 \
    --imc_loss_weight 0.3 \
    --hndc_loss_weight 1.0 
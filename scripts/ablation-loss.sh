export PYTHONPATH=/fgclip/src
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# torchrun --nproc_per_node=4 ./training/main_all_hn_far.py \
python ./training/main_ckpt.py \
    --model ViT-B-32 \
    --pretrained laion400m_e32 \
    --train-data '/fgclip/synthtiger/synthtiger_v1.0_data/synthtiger_v1.0/gt_hn_4.json' \
    --train-mode hn_far \
    --dataset-type clip_json_da \
    --batch-size 128 \
    --precision amp \
    --workers 8 \
    --epochs 3 \
    --cmr_loss_weight 0.2 \
    --imc_loss_weight 0.3 \
    --hndc_loss_weight 1.0 \
    --no_use_hndc_loss \
    --no_use_cmr_loss \
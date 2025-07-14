export PYTHONPATH=/home/work/hard_negative_ocr/code/neg_clip/src
export CUDA_VISIBLE_DEVICES=1

# torchrun --nproc_per_node=4 ./training/main_all_hn_far.py \
python ./training/main_ckpt_finetune_siglip.py \
    --model ViT-L-14 \
    --pretrained openai \
    --train-data '/home/work/cc_ocr/synthtiger/synthtiger_v1.0_data/synthtiger_v1.0/gt_hn_4.json' \
    --train-mode "None_neg" \
    --dataset-type siglip_json_da \
    --batch-size 128 \
    --precision amp \
    --workers 8 \
    --epochs 3 \
    --no_use_cmr_loss \
    --no_use_imc_loss \
    --no_use_hndc_loss \ 
    

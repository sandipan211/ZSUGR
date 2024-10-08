CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4  python ../main.py \
    --image_folder raw_dataset_caddy \
    --method clip_linear_probe \
    --split_number 5 \
    --split_type random \
    --gated_activation gelu \
    --batch_size 16 \
    --epochs 50 \
    --lr_drop 10 \
    --lr 1e-5 \
    --num_cross_attention_layers 3 \
    --setting ablation_clip_only \
    --resume checkpoint_last.pth \
    --best_ckpt checkpoint_best.pth \
    --with_clip_label \
    --fix_clip_label \
    --fix_clip \
    --gzsl
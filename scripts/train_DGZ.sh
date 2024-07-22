CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4  python ../main.py \
    --image_folder raw_dataset_caddy \
    --method existing_zsl \
    --existing_zsl_type DGZ \
    --split_number 1 \
    --split_type random \
    --batch_size 16 \
    --epochs 50 \
    --setting updated_netM \
    --class_embedding 'sent' \
    --attSize 512 \
    --lr 1e-4 \
    --lambda_1 2 \
    --syn_num 50 \
    --nz0 512 \
    --manualSeed 70


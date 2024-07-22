CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4  python ../main.py --image_folder raw_dataset_caddy --method existing_zsl \
    --existing_zsl_type tfvaegan --dataset CADDY --split_type random \
    --split_number 1 --setting bs16_new --gammaD 10 --gammaG 10 \
    --gzsl --manualSeed 3483 --encoded_noise --preprocessing --cuda --image_embedding res101 --class_embedding att \
    --epochs 50 --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 \
    --num_classes 16 --batch_size 16 --nz 512 --latent_size 512 --attSize 512 --resSize 2048 --syn_num 300 \
    --recons_weight 0.01 --a1 1 --a2 1 --feed_lr 0.00001 --dec_lr 0.0001 --feedback_loop 2
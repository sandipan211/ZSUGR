OMP_NUM_THREADS=8  python ../main.py --image_folder raw_dataset_caddy --method existing_zsl \
    --existing_zsl_type FREE --dataset CADDY --split_type random \
    --split_number 5 --setting bs16 --gammaD 10 --gammaG 10 \
    --gzsl --manualSeed 3483 --encoded_noise --preprocessing --cuda --image_embedding res101 --class_embedding att \
    --ngh 4096 --ndh 4096 --lr 0.0001 --classifier_lr 0.001 --lambda1 10 --critic_iter 1 --dataroot data --dataset CADDY \
    --a1 1 --a2 1 --feed_lr 0.00001 --dec_lr 0.0001 --loop 2 \
    --nclass_seen 10 --batch_size 16 --nz 512 --latent_size 512 --attSize 512 --resSize 2048  \
    --syn_num 700 --center_margin 200 --center_weight 0.5 --recons_weight 0.001 --incenter_weight 0.8
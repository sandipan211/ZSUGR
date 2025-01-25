CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4  python ../main.py --image_folder raw_dataset_caddy --method ours \
    --our_method_type GAN --dataset CADDY --split_type random \
    --split_number 1 --resume checkpoint_last.pth \
    --best_ckpt checkpoint_best.pth \
    --gzsl --cuda \
    --epochs 50 --ngh 4096 --ndh 4096 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 \
    --num_classes 16 --batch_size 16 --syn_num 50000 \
    --setting lr_1e-4_transGAN

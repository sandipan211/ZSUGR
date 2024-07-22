CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4  python ../main.py --image_folder raw_dataset_caddy --method ours \
    --our_method_type GAN --dataset CADDY --split_type random \
    --split_number 5 --resume checkpoint_last.pth \
    --best_ckpt checkpoint_best.pth \
    --gzsl --cuda --use_resnet --resSize 2048 \
    --epochs 50 --ngh 4096 --ndh 4096 --classifier_lr 0.001 --lambda1 10 --critic_iter 5 \
    --num_classes 16 --batch_size 16 --syn_num 50000 \
    --setting ablation_resnet_features
python CE_GZSL.py --dataset CADDY --class_embedding sent \
    --syn_num 100 --batch_size 16 --attSize 512 --nz 1024 \
    --embedSize 2048 --outzSize 512 --nhF 2048 --ins_weight 0.001 \
    --cls_weight 0.001 --ins_temp 0.1 --cls_temp 0.1 --manualSeed 3483 \
    --nclass_all 16 --nclass_seen 10 --split_type random --split_number 5 --gzsl True \
    --setting random_5

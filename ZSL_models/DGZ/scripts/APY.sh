#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python ../main.py \
--dataset APY \
--class_embedding 'att' \
--attSize 64 \
--nz0 64 \
--lr 1e-4 \
--syn_num 50 \
--lambda_1 0.04 \



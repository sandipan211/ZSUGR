#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python ../main.py \
--dataset CADDY \
--class_embedding 'sent' \
--attSize 512 \
--nz0 512 \
--lr 1e-4 \
--syn_num 50 \
--lambda_1 2 \



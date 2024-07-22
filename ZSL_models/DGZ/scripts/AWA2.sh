#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python ../main.py \
--dataset AWA2 \
--class_embedding 'att' \
--attSize 85 \
--nz0 85 \
--lr 1e-4 \
--syn_num 100 \
--lambda_1 0.005 \



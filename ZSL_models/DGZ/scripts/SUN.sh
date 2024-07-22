#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python ../main.py \
--dataset SUN \
--class_embedding 'att' \
--attSize 102 \
--nz0 102 \
--lr 1e-4 \
--syn_num 50 \
--lambda_1 0.8 \


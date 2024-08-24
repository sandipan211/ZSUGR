import torch
import torch.nn as nn
import os
import argparse
import time
import h5py
from torch.optim import Adam
from VisionTransformer.custom import VisionTransformer, CLIPClassifier
from VisionTransformer.logger import create_logger
import clip
from models import network as gcat_network
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler.cosine_lr import CosineLRScheduler
import util.trainval as TV
from util import train_gan   
import numpy as np
from torchvision import datasets, transforms
import method_runner

def extract_features(args, dataloaders):
    if args.method == "preprocessing":
        save_path_train = f'/workspace/arijit/sandipan/zsgr_caddy/hariansh/data/CADDY/{args.split_type}_{args.split_number}/gcat_features_{args.setting}.hdf5' 
    elif args.method == "preprocessing_clip":
        save_path_train = f'/workspace/arijit/sandipan/zsgr_caddy/hariansh/data/CADDY/{args.split_type}_{args.split_number}/clip_cls_features_ablation_clip_cgan.hdf5' 

    sentence_features = method_runner.get_sentence_embeddings()
    dataset_loader_train = dataloaders['train']
    dataset_loader_test_seen = dataloaders['test_seen']
    dataset_loader_test_unseen = dataloaders['test_unseen']
    # dir = args.dirs[f'{args.our_method_type}_{args.split_type}_{args.split_number}']
    # os.path.join(args.root, args.output_dir)
    offline_gcat_ckpt_path = os.path.join(args.root, args.output_dir, 'ours', args.our_method_type, args.split_type + '_' + str(args.split_number), args.setting+'_'+args.best_ckpt)
    print(offline_gcat_ckpt_path)
    # offline_gcat_ckpt_path = '/workspace/arijit/sandipan/zsgr_caddy/hariansh/output_dir/ours/GCAT/random_2/lr_1e-5_3dec_withLN_checkpoint_best.pth'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    gcat_model = gcat_network.build(args, sentence_features).to(device)
    offline_gcat_checkpoint = torch.load(offline_gcat_ckpt_path, map_location='cpu')
    print(f'Loaded checkpoint from {offline_gcat_ckpt_path}')
    gcat_model.load_state_dict(offline_gcat_checkpoint['model'])

    gcat_model.eval()
    with torch.no_grad():
        all_features_train = []
        labels_train = []
        for inputs, targets in (dataset_loader_train):
            inputs = inputs.to(device)
            labels = targets['label']
            clip_input = targets['clip_inputs'].to(device)
            features = gcat_model(inputs,is_training=False, clip_input=clip_input)
            if args.method == "preprocessing":
                features = features['gcat_feature']
            elif args.method == "preprocessing_clip":
                features = features['clip_cls_feature']
            features = torch.squeeze(features)
            all_features_train.append(features.cpu().numpy())
            labels_train.append(labels.cpu().numpy())
        all_features_train = np.concatenate(all_features_train, axis=0)
        labels_train = np.concatenate(labels_train,axis=0)

    with torch.no_grad():
        all_features_test_seen = []
        labels_test_seen = []
        for inputs, targets in (dataset_loader_test_seen):
            inputs = inputs.to(device)
            labels = targets['label']
            clip_input = targets['clip_inputs'].to(device)
            features = gcat_model(inputs,is_training=False, clip_input=clip_input)
            if args.method == "preprocessing":
                features = features['gcat_feature']
            elif args.method == "preprocessing_clip":
                features = features['clip_cls_feature']
            features = torch.squeeze(features)
            all_features_test_seen.append(features.cpu().numpy())
            labels_test_seen.append(labels.cpu().numpy())
        all_features_test_seen = np.concatenate(all_features_test_seen, axis=0)
        labels_test_seen = np.concatenate(labels_test_seen,axis=0)

    with torch.no_grad():
        all_features_test_unseen = []
        labels_test_unseen = []
        for inputs, targets in (dataset_loader_test_unseen):
            inputs = inputs.to(device)
            labels = targets['label']
            clip_input = targets['clip_inputs'].to(device)
            features = gcat_model(inputs,is_training=False, clip_input=clip_input)
            if args.method == "preprocessing":
                features = features['gcat_feature']
            elif args.method == "preprocessing_clip":
                features = features['clip_cls_feature']
            features = torch.squeeze(features)
            all_features_test_unseen.append(features.cpu().numpy())
            labels_test_unseen.append(labels.cpu().numpy())
        all_features_test_unseen = np.concatenate(all_features_test_unseen, axis=0)
        labels_test_unseen = np.concatenate(labels_test_unseen,axis=0)
    
    compression = 'gzip' if args.compression else None 
    f = h5py.File(save_path_train, 'w')
    f.create_dataset('feature_map_train', data=all_features_train,compression=compression)
    f.create_dataset('feature_map_test_seen', data=all_features_test_seen,compression=compression)
    f.create_dataset('feature_map_test_unseen', data=all_features_test_unseen,compression=compression)
    f.create_dataset('labels_train', data=labels_train,compression=compression)
    f.create_dataset('labels_test_seen', data=labels_test_seen,compression=compression)
    f.create_dataset('labels_test_unseen', data=labels_test_unseen,compression=compression)
    f.close()

# --dataset CADDY --
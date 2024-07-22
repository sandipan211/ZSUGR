import pickle
import h5py
import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import scipy.io as sio
import torchvision.models.resnet as models
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from torch.utils.data import ConcatDataset
import clip

class CustomedDataset(Dataset):
    def __init__(self, img_set, img_folder, anno_file, transforms=None):
        self.img_set = img_set
        self.img_folder = img_folder
        self.anno_file = anno_file

        self.img_labels = pd.read_csv(self.anno_file)
        # self.clip_feats_folder = clip_feats_folder
        self.transform = transforms
        if self.img_set == 'train':
            print("train dataset object created")
        else:
            print(" test dataset object created")
        # Add a new attribute to store image paths
        self.image_paths = [os.path.join(self.img_folder, img) for img in self.img_labels.iloc[:, 0]]
        device = "cuda" if torch.cuda.is_available() else "cpu"

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # return image
        img_path = os.path.join(self.img_folder, self.img_labels.iloc[idx, 0][1:])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.img_labels.iloc[idx, 1]
        return label, img

def build(image_set, args, transforms):
    root = Path(args.root)
    assert root.exists(), f'provided data path {root} does not exist'
    PATHS = {
        'train': (root / args.image_folder, args.train_path),
        'test_seen': (root / args.image_folder, args.test_seen_path),
        'test_unseen': (root / args.image_folder, args.test_unseen_path)
    }

    # img_folder, anno_file, clip_feats_folder = PATHS[image_set]
    img_folder, anno_file = PATHS[image_set]
    # print(img_folder)


    dataset = CustomedDataset(image_set, img_folder, anno_file, transforms)

    return dataset   

def get_sentence_embeddings():
    # Define label dictionary and sentence template
    label_dict = {
        0: "start comm",
        1: "end comm",
        2: "up",
        3: "down",
        4: "photo",
        5: "backwards",
        6: "carry",
        7: "boat",
        8: "here",
        9: "mosaic",
        10: "num delimiter",
        11: "one",
        12: "two",
        13: "three",
        14: "four",
        15: "five"
    }

    sentence_template = "A photo of a diver gesturing {}"

    # Create dictionary to store sentence features
    sentence_features = {}

    # Load the CLIP model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clip_model, _ = clip.load('ViT-B/32', device=device)   # take from arguments

    # Extract features for each sentence
    for idx, label_name in label_dict.items():
        sentence = sentence_template.format(label_name)
        
        # Tokenize the sentence
        tokenized_text = clip.tokenize([sentence])
        
        # Get the features (embeddings) for the sentence
        with torch.no_grad():
            features = clip_model.encode_text(tokenized_text.to(device=device))
        
        # Store the features for the sentence
        sentence_features[idx] = features

    # Stack features into a single tensor
    sentence_features_tensor = torch.cat(list(sentence_features.values()), dim=0)

    return sentence_features_tensor.float().detach()

def extract_features(config):  # start here

    img_dir = f'data/{config.dataset}'
    file_paths = f'data/xlsa17/data/{config.dataset}/res101.mat'
    save_path_train = f"/workspace/arijit/sandipan/zsgr_caddy/hariansh/data/CADDY/{config.split_type}_{config.split_number}/feature_map_ResNet_101_CADDY_2048.hdf5"
    # save_path_train = '/workspace/arijit/sandipan/zsgr_caddy/hariansh/MTP/data_mtp/CADDY/random_1/abcd.hdf5'

    # save_path_test_seen = f'data/{config.dataset}/feature_map_ResNet_101_{config.dataset}_test_seen.hdf5'
    # save_path_test_unseen = f'data/{config.dataset}/feature_map_ResNet_101_{config.dataset}_test_unseen.hdf5'
    # attribute_path = f'w2v/{config.dataset}_attribute.pkl'

    # region feature extractor
    resnet101 = models.resnet101(pretrained=True).to(config.device)
    resnet101 = nn.Sequential(*list(resnet101.children())[:-1]).eval() # FOR tfvaegan 

    data_transforms = transforms.Compose([
        transforms.Resize(448),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    Dataset_train = build('train', config, data_transforms)
    Dataset_test_seen = build('test_seen', config, data_transforms)
    Dataset_test_unseen = build('test_unseen', config, data_transforms)

    dataset_loader_train = torch.utils.data.DataLoader(Dataset_train,
                                                 batch_size=config.batch_size,
                                                 shuffle=False,
                                                 num_workers=config.nun_workers)
    dataset_loader_test_seen = torch.utils.data.DataLoader(Dataset_test_seen,
                                                 batch_size=config.batch_size,
                                                 shuffle=False,
                                                 num_workers=config.nun_workers)
    dataset_loader_test_unseen = torch.utils.data.DataLoader(Dataset_test_unseen,
                                                 batch_size=config.batch_size,
                                                 shuffle=False,
                                                 num_workers=config.nun_workers)

    with torch.no_grad():
        all_features_train = []
        labels_train = []
        for label, imgs in (dataset_loader_train):
            imgs = imgs.to(config.device)
            features = resnet101(imgs)
            features = torch.squeeze(features)
            if len(features.shape) == 1:  # Check if features is a 1D array
                features = features.unsqueeze(0)  # Reshape to (1, 2048)
            all_features_train.append(features.cpu().numpy())
            labels_train.append(label.cpu().numpy())
        all_features_train = np.concatenate(all_features_train, axis=0)
        labels_train = np.concatenate(labels_train,axis=0)
    with torch.no_grad():
        all_features_test_seen = []
        labels_test_seen = []
        for label, imgs in (dataset_loader_test_seen):
            imgs = imgs.to(config.device)
            features = resnet101(imgs)
            features = torch.squeeze(features)
            if len(features.shape) == 1:  # Check if features is a 1D array
                features = features.unsqueeze(0)  # Reshape to (1, 2048)
            all_features_test_seen.append(features.cpu().numpy())
            labels_test_seen.append(label.cpu().numpy())
        all_features_test_seen = np.concatenate(all_features_test_seen, axis=0)
        labels_test_seen = np.concatenate(labels_test_seen,axis=0)
    with torch.no_grad():
        all_features_test_unseen = []
        labels_test_unseen = []
        for label, imgs in (dataset_loader_test_unseen):
            imgs = imgs.to(config.device)
            features = resnet101(imgs)
            features = torch.squeeze(features)
            if len(features.shape) == 1:  # Check if features is a 1D array
                features = features.unsqueeze(0)  # Reshape to (1, 2048)
            all_features_test_unseen.append(features.cpu().numpy())
            labels_test_unseen.append(label.cpu().numpy())
        all_features_test_unseen = np.concatenate(all_features_test_unseen, axis=0)
        labels_test_unseen = np.concatenate(labels_test_unseen,axis=0)
    # get remaining metadata
    # matcontent = Dataset.matcontent
    # labels = matcontent['labels'].astype(int).squeeze() - 1

    # split_path = os.path.join(f'data/xlsa17/data/{config.dataset}/att_splits.mat')
    # matcontent = sio.loadmat(split_path)
    # trainval_loc = matcontent['trainval_loc'].squeeze() - 1
    # # train_loc = matcontent['train_loc'].squeeze() - 1
    # # val_unseen_loc = matcontent['val_loc'].squeeze() - 1
    # test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
    # test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
    # att = matcontent['att'].T
    # original_att = matcontent['original_att'].T

    # construct attribute w2v
    # with open(attribute_path,'rb') as f:
    #     w2v_att = pickle.load(f)
    # if config.dataset == 'CUB':
    #     assert w2v_att.shape == (312,300)
    # elif config.dataset == 'SUN':
    #     assert w2v_att.shape == (102,300)
    # elif config.dataset == 'AWA2':
    #     assert w2v_att.shape == (85,300)
    w2v_att = get_sentence_embeddings().cpu().numpy()

    compression = 'gzip' if config.compression else None 
    f = h5py.File(save_path_train, 'w')
    f.create_dataset('feature_map_train', data=all_features_train,compression=compression)
    f.create_dataset('feature_map_test_seen', data=all_features_test_seen,compression=compression)
    f.create_dataset('feature_map_test_unseen', data=all_features_test_unseen,compression=compression)
    f.create_dataset('labels_train', data=labels_train,compression=compression)
    f.create_dataset('labels_test_seen', data=labels_test_seen,compression=compression)
    f.create_dataset('labels_test_unseen', data=labels_test_unseen,compression=compression)
    # f.create_dataset('trainval_loc', data=trainval_loc,compression=compression)
    # # f.create_dataset('train_loc', data=train_loc,compression=compression)
    # # f.create_dataset('val_unseen_loc', data=val_unseen_loc,compression=compression)
    # f.create_dataset('test_seen_loc', data=test_seen_loc,compression=compression)
    # f.create_dataset('test_unseen_loc', data=test_unseen_loc,compression=compression)
    # f.create_dataset('att', data=att,compression=compression)
    # f.create_dataset('original_att', data=original_att,compression=compression)
    f.create_dataset('w2v_att', data=w2v_att,compression=compression)
    f.close()



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='CADDY')
    parser.add_argument('--compression', '-c', action='store_true', default=False)
    # parser.add_argument('--batch_size', '-b', type=int, default=200)
    parser.add_argument('--device', '-g', type=str, default='cuda:0')
    parser.add_argument('--nun_workers', '-n', type=int, default='16')
    parser.add_argument('--root', default='/workspace/arijit/sandipan/zsgr_caddy/hariansh', type=str, help='Root directory')
    parser.add_argument('--image_folder', required=True, type=str)#
    parser.add_argument('--enhancement_path',default="a", type=str)
    parser.add_argument('--split_path', default='splits', type=str)
    parser.add_argument('--split_type', default='NF', type=str)
    parser.add_argument('--split_number', default=1, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--num_classes', default=16, type=int)
    parser.add_argument('--use_enhanced_imgs', action='store_true')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--clip_version', default='ViT-B/32', type=str)
    parser.add_argument('--output_dir', default='output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--method', default='ours', help='ours, clip_linear_probe, pretrained_cnn, existing_zsl')
    config = parser.parse_args()
    train_split_name = 'train_' + config.split_type + '_' + str(config.split_number) 
    seen_split_name = 'test_seen_' + config.split_type + '_' + str(config.split_number) 
    unseen_split_name = 'test_unseen_' + config.split_type + '_' + str(config.split_number) 
    split_path = '/workspace/arijit/sandipan/zsgr_caddy/hariansh/raw_dataset_caddy/splits'
    train_path = split_path + '/' + train_split_name + '.csv'
    test_seen_path = split_path + '/' + seen_split_name + '.csv'
    test_unseen_path = split_path + '/' + unseen_split_name + '.csv'
    config.train_path = train_path
    config.test_seen_path=test_seen_path
    config.test_unseen_path = test_unseen_path
    extract_features(config)

# --dataset CADDY --
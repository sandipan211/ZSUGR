
import numpy as np; np.random.seed(1)
import torch; torch.manual_seed(1)
import torch.nn as nn
import torch.nn.functional as F
from time import time
from tqdm import tqdm
from scipy import io
from torch.utils.data import DataLoader
import os
import h5py
import clip
import argparse

DATASET = 'CADDY' # One of ["AWA1", "AWA2", "APY", "CUB", "SUN"]
USE_CLASS_STANDARTIZATION = True # i.e. equation (9) from the paper
USE_PROPER_INIT = True # i.e. equation (10) from the pape

class ClassStandardization(nn.Module):
    """
    Class Standardization procedure from the paper.
    Conceptually, it is equivalent to nn.BatchNorm1d with affine=False,
    but for some reason nn.BatchNorm1d performs slightly worse.
    """
    def __init__(self, feat_dim: int):
        super().__init__()
        
        self.running_mean = nn.Parameter(torch.zeros(feat_dim), requires_grad=False)
        self.running_var = nn.Parameter(torch.ones(feat_dim), requires_grad=False)
    
    def forward(self, class_feats):
        """
        Input: class_feats of shape [num_classes, feat_dim]
        Output: class_feats (standardized) of shape [num_classes, feat_dim]
        """
        if self.training:
            batch_mean = class_feats.mean(dim=0)
            batch_var = class_feats.var(dim=0)
            
            # Normalizing the batch
            result = (class_feats - batch_mean.unsqueeze(0)) / (batch_var.unsqueeze(0) + 1e-5)
            
            # Updating the running mean/std
            self.running_mean.data = 0.9 * self.running_mean.data + 0.1 * batch_mean.detach()
            self.running_var.data = 0.9 * self.running_var.data + 0.1 * batch_var.detach()
        else:
            # Using accumulated statistics
            # Attention! For the test inference, we cant use batch-wise statistics,
            # only the accumulated ones. Otherwise, it will be quite transductive
            result = (class_feats - self.running_mean.unsqueeze(0)) / (self.running_var.unsqueeze(0) + 1e-5)
        
        return result


class CNZSLModel(nn.Module):
    def __init__(self, attr_dim: int, hid_dim: int, proto_dim: int):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(attr_dim, hid_dim),
            nn.ReLU(),
            
            nn.Linear(hid_dim, hid_dim),
            ClassStandardization(hid_dim) if USE_CLASS_STANDARTIZATION else nn.Identity(),
            nn.ReLU(),
            
            ClassStandardization(hid_dim) if USE_CLASS_STANDARTIZATION else nn.Identity(),
            nn.Linear(hid_dim, proto_dim),
            nn.ReLU(),
        )
        
        if USE_PROPER_INIT:
            weight_var = 1 / (hid_dim * proto_dim)
            b = np.sqrt(3 * weight_var)
            self.model[-2].weight.data.uniform_(-b, b)
        
    def forward(self, x, attrs):
        protos = self.model(attrs)
        x_ns = 5 * x / x.norm(dim=1, keepdim=True) # [batch_size, x_dim]
        protos_ns = 5 * protos / protos.norm(dim=1, keepdim=True) # [num_classes, x_dim]
        logits = x_ns @ protos_ns.t() # [batch_size, num_classes]
        
        return logits
    
def getData(opt):
    opt.datadir = os.path.join(opt.datadir, 'data/{}/'.format(opt.dataset))
    path = os.path.join(opt.datadir, '{}_{}'.format(opt.split_type,opt.split_number))
    path = os.path.join(path, 'feature_map_ResNet_101_{}_2048.hdf5'.format(opt.dataset))

    print('_____')
    print(path)

    hf = h5py.File(path, 'r')
    train_feature = np.array(hf.get('feature_map_train')) # removed T
    test_seen_feature = np.array(hf.get('feature_map_test_seen')) # removed T
    test_unseen_feature = np.array(hf.get('feature_map_test_unseen')) # removed T

    train_label = np.array(hf.get('labels_train'))
    test_seen_label = np.array(hf.get('labels_test_seen'))
    test_unseen_label = np.array(hf.get('labels_test_unseen'))

    return train_feature, test_seen_feature, test_unseen_feature, train_label, test_seen_label, test_unseen_label

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

def map_label(label, classes):
    print(label.size())
    mapped_label = torch.LongTensor(label.size())
    for i in range(len(classes)):
        mapped_label[label==classes[i]] = i    

    return mapped_label 

def train_CNZSL(args):

    f = open(f'{args.split_type}_{args.split_number}.txt','a')
    f.write(f'<=============== Loading data for {DATASET} ===============> \n')
    DEVICE = 'cuda' # Set to 'cpu' if a GPU is not available

    # DATA_DIR = f'./data/xlsa17/data/{DATASET}'
    # data = io.loadmat(f'{DATA_DIR}/res101.mat')
    # attrs_mat = io.loadmat(f'{DATA_DIR}/att_splits.mat')
    # feats = data['features'].T.astype(np.float32)
    # labels = data['labels'].squeeze() - 1 # Using "-1" here and for idx to normalize to 0-index
    # train_idx = attrs_mat['trainval_loc'].squeeze() - 1
    # test_seen_idx = attrs_mat['test_seen_loc'].squeeze() - 1
    # test_unseen_idx = attrs_mat['test_unseen_loc'].squeeze() - 1
    # test_idx = np.array(test_seen_idx.tolist() + test_unseen_idx.tolist())
    # seen_classes = sorted(np.unique(labels[test_seen_idx]))
    # unseen_classes = sorted(np.unique(labels[test_unseen_idx]))

    train_feature, test_seen_feature, test_unseen_feature, train_labels, test_seen_label,\
        test_unseen_label = getData(args)
    seen_classes = sorted(np.unique(test_seen_label))
    unseen_classes = sorted(np.unique(test_unseen_label))
    attrs = get_sentence_embeddings()

    f.write(f'<=============== Preprocessing ===============> \n')
    num_classes = len(seen_classes) + len(unseen_classes)
    seen_mask = np.array([(c in seen_classes) for c in range(num_classes)])
    unseen_mask = np.array([(c in unseen_classes) for c in range(num_classes)])
    # attrs = attrs_mat['att'].T
    # attrs = torch.from_numpy(attrs).to(DEVICE).float()
    attrs = attrs / attrs.norm(dim=1, keepdim=True) * np.sqrt(attrs.shape[1])
    attrs_seen = attrs[seen_mask]
    attrs_unseen = attrs[unseen_mask]
    # train_labels = labels[train_idx]
    # test_labels = labels[test_idx]

    # f.write(seen_mask)
    # f.write(unseen_mask)
    test_labels = np.concatenate((test_seen_label, test_unseen_label))
    # test_labels = test_seen_label
    test_features = np.concatenate((test_seen_feature, test_unseen_feature))
    # test_features = test_seen_feature

    test_seen_idx = [i for i, y in enumerate(test_labels) if y in seen_classes]
    test_unseen_idx = [i for i, y in enumerate(test_labels) if y in unseen_classes]
    # labels_remapped_to_seen = [(seen_classes.index(t) if t in seen_classes else -1) for t in labels]
    test_labels_remapped_seen = [(seen_classes.index(t) if t in seen_classes else -1) for t in test_labels]
    test_labels_remapped_unseen = [(unseen_classes.index(t) if t in unseen_classes else -1) for t in test_labels]
    # ds_train = [(feats[i], labels_remapped_to_seen[i]) for i in train_idx]
    mapped_train_labels = map_label(torch.from_numpy(train_labels), seen_classes)
    mapped_train_labels = mapped_train_labels.cpu().numpy()
    ds_train = [(train_feature[i], int(mapped_train_labels[i])) for i in range(len(train_feature))]
    ds_test = [(test_features[i], test_labels[i]) for i in range(len(test_features))]
    train_dataloader = DataLoader(ds_train, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(ds_test, batch_size=16)

    # class_indices_inside_test = {c: [i for i in range(len(test_idx)) if labels[test_idx[i]] == c] for c in range(num_classes)}
    class_indices_inside_test = {c: [i for i in range(len(test_labels)) if test_labels[i] == c] \
                                for c in range(num_classes)}

    f.write(f'\n<=============== Starting training ===============> \n')
    start_time = time()
    model = CNZSLModel(attrs.shape[1], 1024, train_feature.shape[1]).to(DEVICE)
    optim = torch.optim.Adam(model.model.parameters(), lr=0.0005, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, gamma=0.1, step_size=25)


    for epoch in tqdm(range(50)):
        model.train()
        
        for i, batch in enumerate(train_dataloader):
            feats = torch.from_numpy(np.array(batch[0])).to(DEVICE)
            targets = torch.from_numpy(np.array(batch[1])).to(DEVICE)
            logits = model(feats, attrs[seen_mask])
            loss = F.cross_entropy(logits, targets)
            optim.zero_grad()
            loss.backward()
            optim.step()
            # f.write(loss.item())
        
        scheduler.step()

    f.write(f'Training is done! Took time: {(time() - start_time): .1f} seconds \n')

    model.eval() # Important! Otherwise we would use unseen batch statistics
    logits = [model(x.to(DEVICE), attrs).cpu() for x, _ in test_dataloader]
    logits = torch.cat(logits, dim=0)
    logits[:, seen_mask] *= (1.15 if DATASET != "CUB" else 1.0) # Trading a bit of gzsl-s for a bit of gzsl-u
    preds_gzsl = logits.argmax(dim=1).numpy()
    preds_zsl_s = logits[:, seen_mask].argmax(dim=1).numpy()
    preds_zsl_u = logits[:, ~seen_mask].argmax(dim=1).numpy()
    guessed_zsl_u = (preds_zsl_u == test_labels_remapped_unseen)
    guessed_gzsl = (preds_gzsl == test_labels)
    # f.write(guessed_gzsl)
    zsl_unseen_acc = np.mean([guessed_zsl_u[cls_idx].mean().item() for cls_idx in [class_indices_inside_test[c] for c in unseen_classes]]) 
    gzsl_seen_acc = np.mean([guessed_gzsl[cls_idx].mean().item() for cls_idx in [class_indices_inside_test[c] for c in seen_classes]])
    gzsl_unseen_acc = np.mean([guessed_gzsl[cls_idx].mean().item() for cls_idx in [class_indices_inside_test[c] for c in unseen_classes]])
    gzsl_harmonic = 2 * (gzsl_seen_acc * gzsl_unseen_acc) / (gzsl_seen_acc + gzsl_unseen_acc)

    f.write(f'ZSL-U: {zsl_unseen_acc * 100:.02f}\n')
    f.write(f'GZSL-U: {gzsl_unseen_acc * 100:.02f}\n')
    f.write(f'GZSL-S: {gzsl_seen_acc * 100:.02f}\n')
    f.write(f'GZSL-H: {gzsl_harmonic * 100:.02f}\n')

def main():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--root', default='/workspace/arijit/sandipan/zsgr_caddy/hariansh', type=str, help='Root directory')
    parser.add_argument('--image_folder',default='raw_dataset_caddy', type=str)
    parser.add_argument('--enhancement_path', default='', type=str)
    parser.add_argument('--dataset', default='CADDY', type=str)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--split_path', default='splits', type=str)
    parser.add_argument('--split_type', default='random', type=str)
    parser.add_argument('--split_number', default=4, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--num_classes', default=16, type=int)
    parser.add_argument('--use_enhanced_imgs', action='store_true')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--clip_version', default='ViT-B/32', type=str)
    parser.add_argument('--output_dir', default='output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--method', default='existing_zsl', help='ours, clip_linear_probe, pretrained_cnn, existing_zsl')
    parser.add_argument('--our_method_type', default='base-ViT', help='base-ViT')
    parser.add_argument('--existing_zsl_type', default='ViTZSL', help='tfvaegan')
    parser.add_argument('--pretrained_cnn_type', default='resnet50', help='resnet50, vgg16, resnet101')
    parser.add_argument('--setting', default='final', type=str)
    parser.add_argument('--position_embedding', default='sine', type=str, help='type of positional embedding to use on the image (sine or learned)')
    parser.add_argument('--resume', default='')
    parser.add_argument('--best_ckpt', default='')
    parser.add_argument('--gzsl', action='store_true')
    parser.add_argument("--datadir", type=str,default='/workspace/arijit/sandipan/zsgr_caddy/hariansh/')

    args = parser.parse_args()

    train_CNZSL(args)

if __name__ == '__main__':
    main()
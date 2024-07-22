#import h5py
import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
import h5py
import os
import clip 

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i    

    return mapped_label

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

class DATA_LOADER(object):
    def __init__(self, opt):
        self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0

    def read_matdataset(self, opt):
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
    
        # matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        # feature = matcontent['features'].T
        # label = matcontent['labels'].astype(int).squeeze() - 1
        # matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        # trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        # train_loc = matcontent['train_loc'].squeeze() - 1
        # val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        # test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        # test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1    

        att = np.array(hf.get('w2v_att')) # these two lines are needed
        self.attribute = torch.from_numpy(att).float() # removed T
        # self.attribute = get_sentence_embeddings()
        self.attribute /= self.attribute.pow(2).sum(1).sqrt().unsqueeze(1).expand(self.attribute.size(0),self.attribute.size(1))

        if not opt.validation:
            if opt.preprocessing:
                if opt.standardization:
                    print('standardization...')
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()
                
                print(train_feature.shape)
                _train_feature = scaler.fit_transform(train_feature)
                _test_seen_feature = scaler.transform(test_seen_feature)
                _test_unseen_feature = scaler.transform(test_unseen_feature)
                self.train_feature = torch.from_numpy(_train_feature).float()
                mx = self.train_feature.max()
                self.train_feature.mul_(1/mx)
                self.train_label = torch.from_numpy(train_label).long() 
                self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
                self.test_unseen_feature.mul_(1/mx)
                self.test_unseen_label = torch.from_numpy(test_unseen_label).long() 
                self.test_seen_feature = torch.from_numpy(_test_seen_feature).float() 
                self.test_seen_feature.mul_(1/mx)
                self.test_seen_label = torch.from_numpy(test_seen_label).long()

            else:
                self.train_feature = torch.from_numpy(train_feature).float()
                self.train_label = torch.from_numpy(train_label).long() 
                self.test_unseen_feature = torch.from_numpy(test_unseen_feature).float()
                self.test_unseen_label = torch.from_numpy(test_unseen_label).long() 
                self.test_seen_feature = torch.from_numpy(test_seen_feature).float() 
                self.test_seen_label = torch.from_numpy(test_seen_label).long()
        else:
            self.train_feature = torch.from_numpy(train_feature).float()
            self.train_label = torch.from_numpy(train_label).long()
            self.test_unseen_feature = torch.from_numpy(test_unseen_feature).float()
            self.test_unseen_label = torch.from_numpy(test_unseen_label).long() 
    
        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))


        self.ntrain = self.train_feature.size()[0]
        self.ntest_seen = self.test_seen_feature.size()[0]
        self.ntest_unseen = self.test_unseen_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()
        self.train_mapped_label = map_label(self.train_label, self.seenclasses) 

    def next_seen_batch(self, seen_batch):
        idx = torch.randperm(self.ntrain)[0:seen_batch]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_feature, batch_att

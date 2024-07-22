import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
import torch.nn.functional as F
import h5py
import os

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i
    return mapped_label

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

        att = np.array(hf.get('w2v_att'))
        self.attribute = torch.from_numpy(att).float() # removed T

        # matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        # feature = matcontent['features'].T
        # label = matcontent['labels'].astype(int).squeeze() - 1
        # matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        # # numpy array index starts from 0, matlab starts from 1
        mx =5
        # trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        # test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        # test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
        # self.attribute = torch.from_numpy(matcontent['att'].T).float()

        if opt.preprocessing:
            scaler = preprocessing.MinMaxScaler()
            _train_feature = scaler.fit_transform(train_feature)
            _test_seen_feature = scaler.transform(test_seen_feature)
            _test_unseen_feature = scaler.transform(test_unseen_feature)
            self.train_feature = torch.from_numpy(_train_feature).float()
            self.train_feature=F.normalize(self.train_feature,dim=-1)*mx
            self.train_label = torch.from_numpy(train_label).long()
            self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
            self.test_unseen_feature=F.normalize(self.test_unseen_feature,dim=-1)*mx
            self.test_unseen_label = torch.from_numpy(test_unseen_label).long()
            self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
            self.test_seen_feature=F.normalize(self.test_seen_feature,dim=-1)*mx
            self.test_seen_label = torch.from_numpy(test_seen_label).long()
        else:
            self.train_feature = torch.from_numpy(train_feature).float()
            self.train_label = torch.from_numpy(train_label).long()
            self.test_unseen_feature = torch.from_numpy(test_unseen_feature).float()
            self.test_unseen_label = torch.from_numpy(test_unseen_label).long()
            self.test_seen_feature = torch.from_numpy(test_seen_feature).float()
            self.test_seen_label = torch.from_numpy(test_seen_label).long()
    
        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.ntrain = self.train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()

    def next_batch(self, batch_size):
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        return batch_feature, batch_label
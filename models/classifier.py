#author: akshitac8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
# import datasets_tf.image_util as util
from sklearn.preprocessing import MinMaxScaler 
import sys
import copy
import pdb

class CLASSIFIER:
    # train_Y is interger 
    def __init__(self, _train_X, _train_Y, data, _nclass, _cuda, _lr=0.001, _beta1=0.5, _nepoch=50, _batch_size=16, generalized=True):
        self.train_X =  _train_X.clone() 
        self.train_Y = _train_Y.clone() 
        self.test_seen_feature = data['test_seen_feature'].clone()
        self.test_seen_label = data['test_seen_label'] 
        self.test_unseen_feature = data['test_unseen_feature'].clone()
        self.test_unseen_label = data['test_unseen_label']  
        self.seenclasses = data['seen_classes']
        self.unseenclasses = data['unseen_classes']
        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.nclass = _nclass
        self.input_dim = _train_X.size(1)
        self.cuda = _cuda
        self.model =  LINEAR_LOGSOFTMAX_CLASSIFIER(self.input_dim, self.nclass)
        self.label_dict = {
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

        # self.netDec = netDec
        # if self.netDec:
        #     self.netDec.eval()
        #     self.input_dim = self.input_dim + dec_size
        #     self.input_dim += dec_hidden_size
        #     self.model =  LINEAR_LOGSOFTMAX_CLASSIFIER(self.input_dim, self.nclass)
        #     self.train_X = self.compute_dec_out(self.train_X, self.input_dim)
        #     self.test_unseen_feature = self.compute_dec_out(self.test_unseen_feature, self.input_dim)
        #     self.test_seen_feature = self.compute_dec_out(self.test_seen_feature, self.input_dim)
        self.model.apply(self.weights_init)
        self.criterion = nn.NLLLoss()
        self.input = torch.FloatTensor(_batch_size, self.input_dim) 
        self.label = torch.LongTensor(_batch_size) 
        self.lr = _lr
        self.beta1 = _beta1
        self.optimizer = optim.Adam(self.model.parameters(), lr=_lr, betas=(_beta1, 0.999))
        if self.cuda:
            self.model.cuda()
            self.criterion.cuda()
            self.input = self.input.cuda()
            self.label = self.label.cuda()
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.size()[0]
        if generalized:
            self.acc_seen, self.acc_unseen, self.H, self.epoch, self.best_model_gzsl, self.acc_seen_per_class, self.acc_unseen_per_class = self.fit()
            #print('Final: acc_seen=%.4f, acc_unseen=%.4f, h=%.4f' % (self.acc_seen, self.acc_unseen, self.H))
        else:
            self.acc,self.best_model_czsl, self.acc_per_class_czsl = self.fit_zsl() 
            #print('acc=%.4f' % (self.acc))

    def fit_zsl(self):
        best_acc = 0
        mean_loss = 0
        last_loss_epoch = 1e8 
        best_acc_per_class = {}
        best_model = copy.deepcopy(self.model.state_dict())
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):      
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size) 
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)
                   
                inputv = Variable(self.input)
                labelv = Variable(self.label)
                output = self.model(inputv)
                loss = self.criterion(output, labelv)
                # mean_loss += loss.data[0]
                mean_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                #print('Training classifier loss= ', loss.data[0])
            acc_per_class, acc = self.val(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)
            #print('acc %.4f' % (acc))
            if acc > best_acc:
                best_acc = acc
                best_acc_per_class = acc_per_class
                best_model = copy.deepcopy(self.model.state_dict())
        new_acc_per_class = {self.label_dict[key]: value for key, value in best_acc_per_class.items()}
        return best_acc, best_model, new_acc_per_class

    def fit(self):
        best_H = 0.0
        best_seen = 0.0
        best_unseen = 0.0
        best_acc_seen_per_class = {}
        best_acc_unseen_per_class = {}
        out = []
        best_model = copy.deepcopy(self.model.state_dict())
        # early_stopping = EarlyStopping(patience=20, verbose=True)
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):      
                self.model.zero_grad()
                batch_input, batch_label = self.next_batch(self.batch_size) 
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)

                inputv = Variable(self.input)
                labelv = Variable(self.label)
                output = self.model(inputv)
                loss = self.criterion(output, labelv)
                loss.backward()
                self.optimizer.step()
            acc_seen = 0.0
            acc_unseen = 0.0
            acc_seen, acc_seen_per_class = self.val_gzsl(self.test_seen_feature, self.test_seen_label, self.seenclasses)
            acc_unseen, acc_unseen_per_class = self.val_gzsl(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)
            # f = open("debug.txt",'a')
            # f.write(f"acc_seen: {acc_seen}\n")
            # f.write(f"acc_unseen: {acc_unseen}\n")
            # f.close()
            H = 2*acc_seen*acc_unseen / (acc_seen+acc_unseen)
            if H > best_H:
                best_seen = acc_seen
                best_unseen = acc_unseen
                best_H = H
                best_model = copy.deepcopy(self.model.state_dict())
                best_acc_seen_per_class = acc_seen_per_class
                best_acc_unseen_per_class = acc_unseen_per_class
        new_acc_seen_per_class = {self.label_dict[key]: value for key, value in best_acc_seen_per_class.items()}
        new_acc_unseen_per_class = {self.label_dict[key]: value for key, value in best_acc_unseen_per_class.items()}
        return best_seen, best_unseen, best_H, epoch, best_model, new_acc_seen_per_class, new_acc_unseen_per_class

    def map_label(self, label, classes):
        mapped_label = torch.LongTensor(label.size())
        for i in range(len(classes)):
            mapped_label[label==classes[i]] = i    

        return mapped_label   
    
    def reverse_map_labels(self, mapped_label, classes):
        reverse_label = torch.LongTensor(mapped_label.size())
        for i in range(len(classes)):
            reverse_label[mapped_label == i] = classes[i]
        return reverse_label

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
            
    def next_batch(self, batch_size):
        start = self.index_in_epoch
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
        # the last batch
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.train_X[start:self.ntrain]
                Y_rest_part = self.train_Y[start:self.ntrain]
            # shuffle the data
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_X[start:end]
            Y_new_part = self.train_Y[start:end]
            #print(start, end)
            if rest_num_examples > 0:
                return torch.cat((X_rest_part, X_new_part), 0) , torch.cat((Y_rest_part, Y_new_part), 0)
            else:
                return X_new_part, Y_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            #print(start, end)
            # from index start to index end-1
            return self.train_X[start:end], self.train_Y[start:end]

    @torch.no_grad()
    def val_gzsl(self, test_X, test_label, target_classes): 
        start = 0
        ntest = test_X.size()[0]
        # f = open("debug.txt",'a')

        # f.write(f'{target_classes}\n')
        # f.write(f'{test_label.size()}\n')
        # f.write(f'{test_label}\n')
        # f.close()
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            if self.cuda:
                inputX = Variable(test_X[start:end].cuda())
            else:
                inputX = Variable(test_X[start:end])
            output = self.model(inputX)  
            _, predicted_label[start:end] = torch.max(output.data, 1)
            start = end
        # f.write(f'{predicted_label}\n')
        # f.close()
        acc_per_class = self.compute_per_class_acc_gzsl(test_label, predicted_label, target_classes)
        mean_accuracy = sum(acc_per_class.values()) / len(acc_per_class)
        return mean_accuracy, acc_per_class

    @torch.no_grad()
    def compute_per_class_acc_gzsl(self, test_label, predicted_label, target_classes):
        acc_per_class = {}
        for i in target_classes:
            idx = (test_label == i)
            class_accuracy = torch.sum(test_label[idx] == predicted_label[idx]).float() / torch.sum(idx)
            acc_per_class[i] = class_accuracy.item() 
        return acc_per_class 

    # test_label is integer 
    @torch.no_grad()
    def val(self, test_X, test_label, target_classes): 
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            if self.cuda:
                inputX = Variable(test_X[start:end].cuda())
            else:
                inputX = Variable(test_X[start:end])
            output = self.model(inputX) 
            _, predicted_label[start:end] = torch.max(output.data, 1)
            start = end

        # acc = self.compute_per_class_acc(self.map_label(test_label, target_classes), predicted_label, len(target_classes))
        acc_per_class = self.compute_per_class_acc(test_label, self.reverse_map_labels(predicted_label,target_classes), target_classes)
        mean_accuracy = sum(acc_per_class.values()) / len(acc_per_class)

        return acc_per_class, mean_accuracy

    @torch.no_grad()
    def compute_per_class_acc(self, test_label, predicted_label, target_classes):
        acc_per_class = {}
        for i in target_classes:
            idx = (test_label == i)
            class_accuracy = torch.sum(test_label[idx] == predicted_label[idx]).float() / torch.sum(idx)
            acc_per_class[i] = class_accuracy.item() 
        return acc_per_class 


    def compute_dec_out(self, test_X, new_size):
        start = 0
        ntest = test_X.size()[0]
        new_test_X = torch.zeros(ntest,new_size)
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            if self.cuda:
                inputX = Variable(test_X[start:end].cuda(), volatile=True)
            else:
                inputX = Variable(test_X[start:end], volatile=True)
            feat1 = self.netDec(inputX)
            feat2 = self.netDec.getLayersOutDet()
            new_test_X[start:end] = torch.cat([inputX,feat1,feat2],dim=1).data.cpu()
            start = end
        return new_test_X


class LINEAR_LOGSOFTMAX_CLASSIFIER(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX_CLASSIFIER, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)
    def forward(self, x): 
        o = self.logic(self.fc(x))
        return o

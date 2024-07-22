from __future__ import print_function
from locale import normalize
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
from torch.autograd import Variable
import math
from util import *
import numpy as np
from models import gan
# from models import classifier

class TrainGAN():
    def __init__(self, opt, attributes, f):
                                            #changed
        '''
        CLSWGAN trainer
        Inputs:
            opt -- arguments
            unseenAtt -- embeddings vector of unseen classes
            unseenLabels -- labels of unseen classes
            attributes -- embeddings vector of all classes
        '''
        self.opt = opt


        # classifier
        # self.classifier = classifier(num_classes=opt.nclass_all)
        # self.classifier.cuda()

        ## for the unseen class
        # self.Wu_Labels = np.array([i for i, l in enumerate(unseenLabels)])
        # f.write(f"Wu_Labels {self.Wu_Labels}")
        #### Wu : unseen class embedding  contains 15 classes which can be accessed using wu
        # self.Wu = unseenAtt
        # # #### for seen classes
        # self.Ws_Labels = np.array([i for i, l in enumerate(seen_attr_labels)])
        # f.write(f"Ws_Labels {self.Ws_Labels}")
        # self.Ws = seen_attributes

        # self.ntrain = opt.gan_epoch_budget
        self.ntrain = None
        self.attributes = attributes.cpu().data.numpy()

        f.write(f"# of training samples: {self.ntrain} \n")
        # initialize generator and discriminator
        self.netG = gan.MLP_G(self.opt)
        self.netD = gan.MLP_D(self.opt)
        #############################################################

        if self.opt.cuda and torch.cuda.is_available():
            self.netG = self.netG.cuda()
            self.netD = self.netD.cuda()

        f.write('\n\n#############################################################\n')
        f.write(f'{self.netG}\n')
        f.write(f'{self.netD}\n')
        f.write('\n#############################################################\n\n')

        self.one = torch.tensor(1, dtype=torch.float)
        self.mone = self.one * -1
        self.input_res = torch.FloatTensor(self.opt.batch_size, self.opt.resSize)
        self.input_att = torch.FloatTensor(self.opt.batch_size, self.opt.attSize)
        self.noise = torch.FloatTensor(self.opt.batch_size, self.opt.nz)
        self.input_label = torch.LongTensor(self.opt.batch_size)
        self.noise2 = torch.FloatTensor(self.opt.batch_size, self.opt.nz)
        self.cls_criterion = nn.NLLLoss()
        if self.opt.cuda:
            self.one = self.one.cuda()
            self.mone = self.mone.cuda()
            self.input_res = self.input_res.cuda()
            self.input_att = self.input_att.cuda()
            self.noise = self.noise.cuda()
            self.noise2 = self.noise2.cuda()
            self.input_label = self.input_label.cuda()
            self.cls_criterion = self.cls_criterion.cuda()

        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.opt.gan_lr, betas=(self.opt.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.opt.gan_lr, betas=(self.opt.beta1, 0.999))

    def __call__(self, epoch, train_feature, train_label, device, f):
        """
        Train GAN for one epoch
        Inputs:
            epoch: current epoch
            features: current epoch subset of features
            labels: ground truth labels
        """
        self.epoch = epoch  
        self.train_feature = train_feature
        self.ntrain = len(self.train_feature)
        self.train_label = train_label
        self.device = device
        self.f = f
        # self.pretrain_cls = pretrain_cls
        self.trainEpoch()

   
    def load_checkpoint(self, f):
        checkpoint = torch.load(self.opt.netG)
        self.netG.load_state_dict(checkpoint['state_dict'])
        epoch = checkpoint['epoch']
        self.netD.load_state_dict(torch.load(self.opt.netD)['state_dict'])
        f.write(f"loaded weights from epoch: {epoch} \n{self.opt.netD} \n{self.opt.netG} \n")
        return epoch

    def save_checkpoint(self, state='latest'):
        torch.save({'state_dict': self.netD.state_dict(), 'epoch': self.epoch}, f'{self.opt.outname}/disc_{state}.pth')
        torch.save({'state_dict': self.netG.state_dict(), 'epoch': self.epoch}, f'{self.opt.outname}/gen_{state}.pth')

    def generate_syn_feature(self, labels, attribute, num, no_grad=True):
        """
        generates features
        inputs:
            labels: features labels to generate nx1 n is number of objects 
            attributes: attributes of objects to generate (nxd) d is attribute dimensions
            num: number of features to generate for each object
        returns:
            1) synthesised features 
            2) labels of synthesised  features 
        """

        nclass = len(labels)
        syn_feature = torch.FloatTensor(nclass * num , self.opt.resSize)
        syn_label = torch.LongTensor(nclass*num)

        syn_att = torch.FloatTensor(num, self.opt.attSize)
        syn_noise = torch.FloatTensor(num, self.opt.nz)
        
        if self.opt.cuda:
            syn_att = syn_att.cuda()
            syn_noise = syn_noise.cuda()
        if no_grad is True:
            with torch.no_grad():
                for i in range(nclass):
                    label = labels[i]
                    iclass_att = attribute[i]
                    syn_att.copy_(iclass_att.repeat(num, 1))
                    syn_noise.normal_(0, 1)
                    output = self.netG(Variable(syn_noise), Variable(syn_att))
                
                    syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
                    syn_label.narrow(0, i*num, num).fill_(label)
        else:
            for i in range(nclass):
                label = labels[i]
                iclass_att = attribute[i]
                syn_att.copy_(iclass_att.repeat(num, 1))
                syn_noise.normal_(0, 1)
                output = self.netG(Variable(syn_noise), Variable(syn_att))
            
                syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
                syn_label.narrow(0, i*num, num).fill_(label)

        return syn_feature, syn_label

    def calc_gradient_penalty(self, real_data, fake_data, input_att):
        alpha = torch.rand(self.opt.batch_size, 1)
        alpha = alpha.expand(real_data.size())
        if self.opt.cuda:
            alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        if self.opt.cuda:
            interpolates = interpolates.cuda()

        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = self.netD(interpolates, Variable(input_att))

        ones = torch.ones(disc_interpolates.size())
        if self.opt.cuda:
            ones = ones.cuda()

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                grad_outputs=ones,
                                create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.opt.lambda1
        return gradient_penalty

    def get_z_random(self):
        """
        returns normal initialized noise tensor 
        """
        z = torch.cuda.FloatTensor(self.opt.batch_size, self.opt.nz)
        z.normal_(0, 1)
        return z

    def loss_fn(self, recon_x, x):
        BCE = torch.nn.functional.binary_cross_entropy(recon_x+1e-12, x.detach(),size_average=False)
        BCE = BCE.sum()/ x.size(0)
        # what is this log_var and KLD ???
        # KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())/ x.size(0)
        # return (BCE + KLD)
        return BCE

    def map_label(self, label, classes):
        mapped_label = torch.LongTensor(label.size())
        for i in range(len(classes)):
            mapped_label[label==classes[i]] = i    

        return mapped_label 

    def sample(self):
        """
        randomaly samples one batch of data
        returns (1)real features, (2)labels (3) attributes embeddings
        """
        idx = torch.randperm(self.ntrain)[0:self.opt.batch_size]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attributes[batch_label]
        self.input_res.copy_(torch.tensor(batch_feature))
        self.input_att.copy_(torch.tensor(batch_att))
        self.input_label.copy_(self.map_label(batch_label, self.opt.split_labels['train'])) # why???

    def trainEpoch(self):
        f = self.f

        errG_avg = 0.0
        D_cost_avg = 0.0
        Wasserstein_D_avg = 0.0
        G_cost_avg = 0.0
        loss_lz_avg = 0.0
        # c_errG_avg = 0.0
        self.netG.train()
        for i in range(0, self.ntrain, self.opt.batch_size):
            # p = open("debug.txt",'a')
            # p.write(f'{input_label}\n')
            # p.write(f'{input_att.size()}\n')
            # p.write(f'{input_att}\n')
            # p.close()
            ############################
            # (1) Update D network: optimize WGAN-GP objective, Equation (2)
            ###########################
            for p in self.netD.parameters(): # reset requires_grad
                p.requires_grad = True # they are set to False below in netG update

            for iter_d in range(self.opt.critic_iter):
                self.sample()  
                input_resv = Variable(self.input_res)
                input_attv = Variable(self.input_att)
                self.netD.zero_grad()
                # train with realG
                # sample a mini-batch
                # sparse_real = self.opt.resSize - input_res[1].gt(0).sum()

                criticD_real = self.netD(input_resv, input_attv)
                criticD_real = criticD_real.mean()
                criticD_real.backward(self.mone)

                # train with fakeG
                self.noise.normal_(0, 1)

                noisev = Variable(self.noise)
                fake = self.netG(noisev, input_attv)
                criticD_fake = self.netD(fake.detach(), input_attv)
                criticD_fake = criticD_fake.mean()
                criticD_fake.backward(self.one)

                # gradient penalty
                gradient_penalty = self.calc_gradient_penalty(self.input_res, fake.data, self.input_att)
                gradient_penalty.backward()

                # unseenc_errG.backward()

                Wasserstein_D = criticD_real - criticD_fake
                D_cost = criticD_fake - criticD_real + gradient_penalty
                # D_cost.backward()

                self.optimizerD.step()
                D_cost_avg += D_cost.item()
                Wasserstein_D_avg += Wasserstein_D.item()

            ############################
            # (2) Update G network: optimize WGAN-GP objective, Equation (2)
            ###########################
            for p in self.netD.parameters(): # reset requires_grad
                p.requires_grad = False # avoid computation
            
            self.netG.zero_grad()
            self.noise.normal_(0, 1)
            noisev = Variable(self.noise)
            fake = self.netG(noisev, input_attv)
            criticG_fake = self.netD(fake, input_attv)
            criticG_fake = criticG_fake.mean()
            G_cost = criticG_fake
            # recons_loss = self.loss_fn(fake,input_resv) # temporary

            # Classification cost
            # c_errG = (self.opt.cls_weight)*self.cls_criterion(self.pretrain_cls.model(fake), Variable(self.input_label))

            #mode seeking loss https://github.com/HelenMao/MSGAN/blob/e386c252f059703fcf5d439258949ae03dc46519/DCGAN-Mode-Seeking/model.py#L66
            self.noise2.normal_(0, 1)
            noise2v = Variable(self.noise2)
            fake2 = self.netG(noise2v, input_attv)

            lz = torch.mean(torch.abs(fake2 - fake)) / torch.mean(torch.abs(noise2v - noisev))
            eps = 1 * 1e-5
            loss_lz = 1 / (lz + eps)
            loss_lz*=self.opt.lz_ratio

            # # ---------------------------------------------
            # Total loss 

            errG= -G_cost + loss_lz 
            
            errG.backward()
            self.optimizerG.step()
            
            # f.write(f"[{self.epoch+1:02}/{self.opt.epochs:02}] [{i:06}/{int(self.ntrain)}] \
            # Loss: {errG.item() :0.4f} D loss: {D_cost.item():.4f} G loss: {G_cost.item():.4f}, W dist: {Wasserstein_D.item():.4f} \n")
            errG_avg += errG.item()
            G_cost_avg += G_cost.item()
            loss_lz_avg += loss_lz.item()
            # c_errG_avg += c_errG.item()
        
        num_iter = self.ntrain/self.opt.batch_size
        errG_avg /= num_iter
        loss_lz_avg /= num_iter
        G_cost_avg /= num_iter
        # c_errG_avg /= num_iter
        D_cost_avg /= (num_iter*self.opt.critic_iter)
        Wasserstein_D_avg /= (num_iter*self.opt.critic_iter)
        f.write(f"Epoch: {self.epoch} Loss: {errG_avg :0.4f} D loss: {D_cost_avg:.4f} G loss: {G_cost_avg:.4f}, W dist: {Wasserstein_D_avg:.4f} loss_lz: {loss_lz_avg: .4f} \n")
        self.netG.eval()


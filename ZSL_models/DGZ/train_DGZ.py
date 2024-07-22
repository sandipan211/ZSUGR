from __future__ import print_function
import random
import torch
import torch.autograd as autograd
import torch.optim as optim
import DGZ.util as util
import DGZ.classifier as classifier
import DGZ.model as model
from ZSL_models.TransZero_pp.logger import create_logger
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from config import opt

def train_dgz(opt):
    dir = opt.dirs[f'{opt.existing_zsl_type}_{opt.split_type}_{opt.split_number}']
    logger = create_logger(output_dir=dir, eval=opt.eval, setting_name=opt.setting)
    logger.info(opt)

    # set random seed
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    logger.info(f'Random Seed: {opt.manualSeed}')
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)

    # load data
    data = util.DATA_LOADER(opt)
    # data.train_feature=F.normalize(data.train_feature,dim=1)*5#normalize the visual feature
    logger.info(f'# of training samples: {data.ntrain}')

    # model initialization
    netG = model.Generator(opt.netG_layer_sizes, opt.nz0, opt.attSize).cuda()
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    logger.info(netG)
    netD = model.Discriminator(opt.resSize,opt.attSize,opt.netD_layer_sizes).cuda()
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    logger.info(netD)
    netM=model.Mapping_net(opt.netM_layer_sizes,opt.attSize).cuda()# initialize the mapping net
    logger.info(netM)

    def sample(batch_size=opt.batch_size):
        batch_feature, batch_label= data.next_batch(batch_size)
        return batch_feature.cuda(),batch_label.cuda()

    def calc_gradient_penalty(netD, real_data, fake_data, input_att):

        alpha = torch.rand(opt.batch_size, 1)
        alpha = alpha.expand(real_data.size()).cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = interpolates.cuda()
        interpolates.requires_grad_(True)

        disc_interpolates = netD(interpolates, input_att)
        ones = torch.ones(disc_interpolates.size()).cuda()

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                grad_outputs=ones,
                                create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda_0
        return gradient_penalty

    # generate viusal samples with netG
    def visual_generation(netG, classes, att, num):
        nclass = classes.size(0)
        syn_feature = torch.tensor([])
        syn_label = torch.LongTensor([])
        for i in range(nclass):
            iclass = classes[i]
            iclass_att = att[iclass]
            syn_att = iclass_att.repeat(num, 1).cuda()
            syn_noise = torch.randn(num, opt.nz0).cuda()
            with torch.no_grad():
                output = netG(syn_noise, syn_att).cpu()
            syn_feature=torch.cat((syn_feature,output),dim=0)
            syn_label=torch.cat((syn_label,iclass.repeat(num)))
        return syn_feature, syn_label
    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    #############################
    #generator training phase
    #############################
    best_gzsl_seen_acc=0
    best_gzsl_unseen_acc=0
    best_H=0
    best_model_params_path = dir
    last_chk_point_path = dir
    best_predicted_label_seen = torch.empty(0)
    best_predicted_label_unseen = torch.empty(0)
    for epoch in range(opt.epochs):
        for i in range(0, data.ntrain, opt.batch_size):
            # (1) Update netD
            for p in netD.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update
            for p in netG.parameters():  # reset requires_grad
                p.requires_grad = False # they are set to Ture below in netG update
            for iter_d in range(opt.critic_iter):
                netD.zero_grad()
                batch_res,batch_label=sample(opt.batch_size)
                batch_att=data.attribute.cuda()[batch_label]
                criticD_real = netD(batch_res, batch_att).mean()

                z1 = opt.att_std*torch.randn([opt.batch_size, opt.nz0]).cuda()# noise for attribute augmentation
                as_fake = z1 + batch_att# augmented attribute
                z0=torch.randn(opt.batch_size,opt.nz0).cuda()# noise for generation
                fake = netG(z0, as_fake).detach()
                criticD_fake = netD(fake, batch_att).mean()
                
                gradient_penalty = calc_gradient_penalty(netD, batch_res, fake.data, batch_att)# gradient penalty

                W_D = (criticD_real - criticD_fake).item()
                D_cost = criticD_fake - criticD_real + gradient_penalty# cost of the discriminator
                D_cost.backward()
                optimizerD.step()
                D_cost=D_cost.item()
            # (2) Update netG
            for p in netD.parameters():  # reset requires_grad
                p.requires_grad = False
            for p in netG.parameters():  # reset requires_grad
                p.requires_grad = True # they are set to False above in netD update
            netG.zero_grad()
            fake = netG(z0, as_fake)
            G_cost = -netD(fake, batch_att).mean()
            G_cost.backward()
            optimizerG.step()
            G_cost = G_cost.item()# cost of the generator

        logger.info('[%d/%d] Loss_D: %.4f Loss_G: %.4f W_d: %.4f'
            % (epoch, opt.epochs, D_cost, G_cost, W_D))
        #############################
        # classifier training phase
        #############################
        netG.eval()
        syn_feature, syn_label= visual_generation(netG, data.unseenclasses, data.attribute,opt.syn_num)# generate pseudo unseen samples
        train_X = torch.cat((data.train_feature,syn_feature),dim=0)
        train_Y = torch.cat((data.train_label,syn_label),dim=0)

        cls_gzsl = classifier.CLASSIFIER(opt.netM_layer_sizes,opt.lambda_1, train_X,
                                        train_Y, data,opt.lr_classifier, opt.beta1, opt.nepoch_classifier,opt.batch_size,opt.temperature)

        acu = cls_gzsl.acc_unseen
        acs = cls_gzsl.acc_seen
        ach = cls_gzsl.H
        logger.info('unseen=%.4f, seen=%.4f, h=%.4f' % (acu, acs, ach))
        if(ach > best_H):
            best_H = ach
            best_gzsl_seen_acc = acs
            best_gzsl_unseen_acc = acu
            best_acc_seen_per_class = cls_gzsl.acc_seen_per_class
            best_acc_unseen_per_class = cls_gzsl.acc_unseen_per_class
            best_predicted_label_seen = cls_gzsl.predicted_label_seen
            best_predicted_label_unseen = cls_gzsl.predicted_label_unseen
            torch.save(netG.state_dict(), os.path.join(best_model_params_path,'best_chk_point_netG.pt'))
            torch.save(netD.state_dict(), os.path.join(best_model_params_path,'best_chk_point_netD.pt'))
            torch.save(cls_gzsl.best_gzsl_model, os.path.join(best_model_params_path,'best_chk_point_netM.pt'))

    logger.info('Best GZSL: unseen=%.4f, seen=%.4f, h=%.4f' %(best_gzsl_unseen_acc, best_gzsl_seen_acc, best_H))
    logger.info(f'Class-wise acc seen: {best_acc_seen_per_class}\n')
    logger.info(f'Class-wise acc unseen: {best_acc_unseen_per_class}\n')
    torch.save(best_acc_seen_per_class, os.path.join(best_model_params_path,'best_acc_seen_per_class_gzsl.pt'))
    torch.save(best_acc_unseen_per_class, os.path.join(best_model_params_path,'best_acc_unseen_per_class_gzsl.pt'))
    torch.save(best_predicted_label_seen, os.path.join(best_model_params_path,'best_predicted_label_seen.pt'))
    torch.save(best_predicted_label_unseen, os.path.join(best_model_params_path,'best_predicted_label_unseen.pt'))
    torch.save(netG.state_dict(), os.path.join(last_chk_point_path,'last_chk_point_netG.pt'))
    torch.save(netD.state_dict(), os.path.join(last_chk_point_path,'last_chk_point_netD.pt'))
    torch.save(netM.state_dict(), os.path.join(last_chk_point_path,'last_chk_point_netM.pt'))

    create_confusion_matrix(torch.cat(best_predicted_label_seen, best_predicted_label_unseen), torch.cat(data.test_seen_label, data.test_unseen_label), "confusion_DGZ")

def create_confusion_matrix(predicted_labels, true_labels, filename):

    # Example class names
    class_names = [
        "start comm", "end comm", "up", "down", "photo", "backwards", "carry", "boat", "here",
            "mosaic", "num delimiter", "one", "two", "three", "four", "five"
    ]
    predicted_labels_np = predicted_labels.cpu().numpy()
    true_labels_np = true_labels.cpu().cpu().numpy()
    # Compute confusion matrix with class names
    conf_matrix = confusion_matrix(true_labels_np, predicted_labels_np, labels=np.arange(len(class_names)))

    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    norm_conf_matrix = conf_matrix / row_sums
    # Convert confusion matrix to DataFrame for better visualization
    conf_matrix_df = pd.DataFrame(norm_conf_matrix, index=class_names, columns=class_names)

    plt.figure(figsize=(12, 8))
    sns.heatmap(conf_matrix_df, annot=True, cmap='Blues', fmt='.2f')

    # Rotate x-axis labels
    plt.xticks(rotation=30, ha = 'right')

    plt.xlabel('Predicted Label', fontsize = 20)
    plt.ylabel('True Label', fontsize = 20)
    # plt.title('Confusion Matrix')

    # Save as PNG, EPS, and PDF files
    plt.savefig(filename+'.png', format='png', bbox_inches='tight')
    plt.savefig(filename+'.eps', format='eps', bbox_inches='tight')
    plt.savefig(filename+'.pdf', format='pdf', bbox_inches='tight')

    





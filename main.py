from datasets import build_dataset
from util.split import make_caddy_splits
import argparse
import datetime
import torch
from torch.utils.data import DataLoader, DistributedSampler
import method_runner, method_runner_gan, method_runner_cnn
from VisionTransformer import Clip_test
from VisionTransformer import custom
import random
from pathlib import Path
import logging
import numpy as np
import os
from ZSL_models.TransZero_pp import train_cub
from ZSL_models.tfvaegan import train_images
from ZSL_models.FREE import train_free
from ZSL_models.DGZ import train_DGZ
from ZSL_models.HASZSL import main as train_HASZSL
import clip
import pandas as pd 
from preprocessing_gcat import extract_features

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def create_dir(print_str, path, dir_dict):
    print(f'{print_str}: {path}')
    Path(path).mkdir(parents=True, exist_ok=True)
    key = "_".join(print_str.split()[:-1])
    dir_dict[key] = path

def setup_dirs(args):

    dir_paths = {}
    clip_versions = {
        'ViT-B/32': 'vitb32',
        'ViT-B/16': 'vitb16',
        'ViT-L/14': 'vitl14'
    }

    # make split directory
    split_path = os.path.join(args.root, args.image_folder, args.split_path)
    create_dir('split directory', split_path, dir_paths)
    # make output directory
    op_dir_path = os.path.join(args.root, args.output_dir)
    create_dir('output directory', op_dir_path, dir_paths)
    # make method directory
    if args.method != 'preprocessing' or args.method!= 'preprocessing_clip':
        method_path = os.path.join(op_dir_path, args.method)
        create_dir(f'{args.method} directory', method_path, dir_paths)

    if args.method == 'ours':
        our_method_type_path = os.path.join(method_path, args.our_method_type)
        create_dir(f'{args.our_method_type} directory', our_method_type_path, dir_paths)
        result_path = os.path.join(our_method_type_path, args.split_type + '_' + str(args.split_number))
        create_dir(f'{args.our_method_type} {args.split_type} {args.split_number} directory', result_path, dir_paths)
    elif args.method == 'clip_linear_probe':
        clip_linear_probe_path = os.path.join(method_path, clip_versions[args.clip_version])
        create_dir(f'clip {args.clip_version} linear probe directory', clip_linear_probe_path, dir_paths)
        result_path = os.path.join(clip_linear_probe_path, args.split_type + '_' + str(args.split_number))
        create_dir(f'{args.clip_version} {args.split_type} {args.split_number} directory', result_path, dir_paths)
    elif args.method == 'pretrained_cnn':
        pretrained_cnn_type_path = os.path.join(method_path, args.pretrained_cnn_type)
        create_dir(f'{args.pretrained_cnn_type} directory', pretrained_cnn_type_path, dir_paths)
        result_path = os.path.join(pretrained_cnn_type_path, args.split_type + '_' + str(args.split_number))
        create_dir(f'{args.pretrained_cnn_type} {args.split_type} {args.split_number} directory', result_path, dir_paths)
    elif args.method == 'existing_zsl':
        existing_zsl_type_path = os.path.join(method_path, args.existing_zsl_type)
        create_dir(f'{args.existing_zsl_type} directory', existing_zsl_type_path, dir_paths)
        result_path = os.path.join(existing_zsl_type_path, args.split_type + '_' + str(args.split_number))
        create_dir(f'{args.existing_zsl_type} {args.split_type} {args.split_number} directory', result_path, dir_paths)
    elif args.method == 'preprocessing' or args.method == 'preprocessing_clip':
        pass
    else:
        raise ValueError(f'Method {args.method} not supported')
    
    # print(dir_paths)
    return dir_paths

def get_split_labels(train_path, test_seen_path, test_unseen_path):

    # get the unique 0-indexed labels for the splits
    split_dict = {}

    df = pd.read_csv(train_path)
    split_dict['train'] = sorted(list(df['label id'].unique()))

    df = pd.read_csv(test_seen_path)
    split_dict['test_seen'] = sorted(list(df['label id'].unique()))

    df = pd.read_csv(test_unseen_path)
    split_dict['test_unseen'] = sorted(list(df['label id'].unique()))    

    return split_dict

def get_args_parser_known():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--root', default='/workspace/arijit/sandipan/zsgr_caddy/hariansh', type=str, help='Root directory')
    parser.add_argument('--image_folder',default='raw_dataset_caddy', type=str)
    parser.add_argument('--enhancement_path', default='', type=str)
    parser.add_argument('--dataset', default='CADDY', type=str)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--split_path', default='splits', type=str)
    parser.add_argument('--split_type', default='random', type=str)
    parser.add_argument('--split_number', default=1, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--num_classes', default=16, type=int)
    parser.add_argument('--use_enhanced_imgs', action='store_true')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--clip_version', default='ViT-B/32', type=str)
    parser.add_argument('--output_dir', default='output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--method', default='ours', help='ours, clip_linear_probe, pretrained_cnn, existing_zsl')
    parser.add_argument('--our_method_type', default='base-ViT', help='base-ViT')
    parser.add_argument('--existing_zsl_type', default='tfvaegan', help='tfvaegan')
    parser.add_argument('--pretrained_cnn_type', default='resnet50', help='resnet50, vgg16, resnet101')
    parser.add_argument('--setting', default='final', type=str)
    parser.add_argument('--position_embedding', default='sine', type=str, help='type of positional embedding to use on the image (sine or learned)')
    parser.add_argument('--resume', default='')
    parser.add_argument('--best_ckpt', default='')
    parser.add_argument('--gzsl', action='store_true')
    parser.add_argument("--datadir", type=str,default='/workspace/arijit/sandipan/zsgr_caddy/hariansh/')
    parser.add_argument("--continue_lastbest_performance", action='store_true',default=False)
    
    # clip related
    parser.add_argument('--with_clip_label', action='store_true', help='Use clip to classify gesture')
    parser.add_argument('--fix_clip_label', action='store_true', help='')
    parser.add_argument('--clip_embed_dim', default=512, type=int)
    parser.add_argument('--fix_clip', action='store_true', help='')

    # architecture related
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_drop', default=100, type=int)
    parser.add_argument('--lr_drop_gamma', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--backbone', default='resnet50', type=str, help="Name of the convolutional backbone to use")
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--enc_layers', default=3, type=int)
    parser.add_argument('--cnn_embed_dim', default=256, type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--nhead', default=8, type=int)
    parser.add_argument('--num_cross_attention_layers', default=1, type=int)
    parser.add_argument('--encoder_only', action='store_true',default=False, help='Use only encoder as feature extractor')
    parser.add_argument('--seed',type=int,default=233)
    parser.add_argument('--compression',action='store_true',default=True)
    parser.add_argument('--gated_activation', type=str,default="gelu")

    args, remaining_args = parser.parse_known_args()
    return args, remaining_args, parser

def args_GAN(parser, remaining_args, args):
    parser.add_argument('--syn_num', type=int, default=100, help='number features to generate per class')
    parser.add_argument('--resSize', type=int, default=512, help='size of visual features')
    parser.add_argument('--attSize', type=int, default=512, help='size of semantic features')
    parser.add_argument('--nz', type=int, default=512, help='size of the latent z vector')
    parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
    parser.add_argument('--ndh', type=int, default=1024, help='size of the hidden units in discriminator')
    parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
    parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
    parser.add_argument('--lambda2', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
    parser.add_argument('--gan_lr', type=float, default=0.0001, help='learning rate to train GANs ')
    parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument("--latent_size", type=int, default=512)
    parser.add_argument('--lz_ratio', type=float, default=0.0001)
    parser.add_argument('--pretrain_classifier',type=str,default='')
    parser.add_argument('--cls_weight',type=float,default=0.001)
    parser.add_argument('--use_resnet',action='store_true',default=False, help='Use ResNet Features')
    parser.add_argument('--use_gcat', action='store_true', help='enbale MinMaxScaler on visual features')
    parser.add_argument('--use_clip', action='store_true', help='enbale MinMaxScaler on visual features')

    ###

    args = parser.parse_args(remaining_args, namespace=args)
    args.lambda2 = args.lambda1
    args.latent_size = args.attSize

    return args

def args_tfvaegan(parser, remaining_args,args):
    parser.add_argument('--image_embedding', default='res101')
    parser.add_argument('--class_embedding', default='att')
    parser.add_argument('--syn_num', type=int, default=100, help='number features to generate per class')
    # parser.add_argument('--gzsl', action='store_true', default=False, help='enable generalized zero-shot learning')
    parser.add_argument('--preprocessing', action='store_true', default=False, help='enbale MinMaxScaler on visual features')
    parser.add_argument('--standardization', action='store_true', default=False)
    parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
    parser.add_argument('--attSize', type=int, default=1024, help='size of semantic features')
    parser.add_argument('--nz', type=int, default=312, help='size of the latent z vector')
    parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
    parser.add_argument('--ndh', type=int, default=1024, help='size of the hidden units in discriminator')
    parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
    parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
    parser.add_argument('--lambda2', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
    # parser.add_argument('--lr', type=float, default=0.001, help='learning rate to train GANs ')
    parser.add_argument('--feed_lr', type=float, default=0.0001, help='learning rate to train GANs ')
    parser.add_argument('--dec_lr', type=float, default=0.0001, help='learning rate to train GANs ')
    parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
    parser.add_argument('--encoded_noise', action='store_true', default=False, help='enables validation mode')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--validation', action='store_true', default=False, help='enables validation mode')
    parser.add_argument("--encoder_layer_sizes", type=list, default=[8192, 4096])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[4096, 8192])
    parser.add_argument('--gammaD', type=int, default=1000, help='weight on the W-GAN loss')
    parser.add_argument('--gammaG', type=int, default=1000, help='weight on the W-GAN loss')
    parser.add_argument('--gammaG_D2', type=int, default=1000, help='weight on the W-GAN loss')
    parser.add_argument('--gammaD2', type=int, default=1000, help='weight on the W-GAN loss')
    parser.add_argument("--latent_size", type=int, default=312)
    parser.add_argument("--conditional", action='store_true',default=True)
    # parser.add_argument("--datadir", type=str,default='/workspace/arijit/sandipan/zsgr_caddy/hariansh/')
    ###

    parser.add_argument('--a1', type=float, default=1.0)
    parser.add_argument('--a2', type=float, default=1.0)
    parser.add_argument('--recons_weight', type=float, default=1.0, help='recons_weight for decoder')
    parser.add_argument('--feedback_loop', type=int, default=2)
    parser.add_argument('--freeze_dec', action='store_true', default=False, help='Freeze Decoder for fake samples')

    args = parser.parse_args(remaining_args, namespace=args)
    args.lambda2 = args.lambda1
    args.encoder_layer_sizes[0] = args.resSize
    args.decoder_layer_sizes[-1] = args.resSize
    args.latent_size = args.attSize

    return args

def args_free(parser, remaining_args,args):
    parser.add_argument('--dataroot', default='data', help='path to dataset')
    parser.add_argument('--image_embedding', default='res101')
    parser.add_argument('--class_embedding', default='att')
    parser.add_argument('--syn_num', type=int, default=100, help='number features to generate per class')
    # parser.add_argument('--gzsl', action='store_true', default=False, help='enable generalized zero-shot learning')
    parser.add_argument('--preprocessing', action='store_true', default=False, help='enbale MinMaxScaler on visual features')
    parser.add_argument('--standardization', action='store_true', default=False)
    parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
    parser.add_argument('--attSize', type=int, default=1024, help='size of semantic features')
    parser.add_argument('--nz', type=int, default=312, help='size of the latent z vector')
    parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
    parser.add_argument('--ndh', type=int, default=1024, help='size of the hidden units in discriminator')
    parser.add_argument('--cls_nepoch', type=int, default=2000, help='number of epochs to train for classifier')
    parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
    parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
    parser.add_argument('--lambda2', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
    # parser.add_argument('--lr', type=float, default=0.001, help='learning rate to train GANs ')
    parser.add_argument('--feed_lr', type=float, default=0.0001, help='learning rate to train GANs ')
    parser.add_argument('--dec_lr', type=float, default=0.0001, help='learning rate to train GANs ')
    parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
    parser.add_argument('--encoded_noise', action='store_true', default=False, help='enables validation mode')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')
    parser.add_argument('--validation', action='store_true', default=False, help='enables validation mode')
    parser.add_argument("--encoder_layer_sizes", type=list, default=[8192, 4096])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[4096, 8192])
    parser.add_argument('--gammaD', type=int, default=1000, help='weight on the W-GAN loss')
    parser.add_argument('--gammaG', type=int, default=1000, help='weight on the W-GAN loss')
    parser.add_argument('--gammaG_D2', type=int, default=1000, help='weight on the W-GAN loss')
    parser.add_argument('--gammaD2', type=int, default=1000, help='weight on the W-GAN loss')
    parser.add_argument("--latent_size", type=int, default=312)
    parser.add_argument("--conditional", action='store_true',default=True)
    # parser.add_argument("--datadir", type=str,default='/workspace/arijit/sandipan/zsgr_caddy/hariansh/')
    ###

    parser.add_argument('--a1', type=float, default=1.0)
    parser.add_argument('--a2', type=float, default=1.0)
    parser.add_argument('--recons_weight', type=float, default=0.01, help='recons_weight for decoder')
    parser.add_argument('--loop', type=int, default=2)
    parser.add_argument('--freeze_dec', action='store_true', default=False, help='Freeze Decoder for fake samples')
    #############################################################
    parser.add_argument('--result_root', type=str, default='/workspace/arijit/sandipan/zsgr_caddy/hariansh/ZSL_models/FREE', help='root path for saving checkpoint')
    parser.add_argument('--center_margin', type=float, default=150, help='the margin in the center loss')
    parser.add_argument('--center_weight', type=float, default=0.5, help='the weight for the center loss')
    parser.add_argument('--incenter_weight', type=float, default=0.5, help='the weight for the center loss')
    parser.add_argument('--cls_weight', type=float, default=0.2, help='weight of the classification loss')
    parser.add_argument('--nclass_seen', type=int, default=150, help='number of seen classes')
    parser.add_argument('--latensize', type=int, default=2048, help='size of semantic features')
    parser.add_argument('--i_c', type=float, default=0.1, help='information constrain')
    parser.add_argument('--lr_dec', action='store_true', default=False, help='enable lr decay or not')
    parser.add_argument('--lr_dec_ep', type=int, default=1, help='lr decay for every 100 epoch')

    args = parser.parse_args(remaining_args, namespace=args)
    args.lambda2 = args.lambda1
    args.encoder_layer_sizes[0] = args.resSize
    args.decoder_layer_sizes[-1] = args.resSize
    args.latent_size = args.attSize

    return args

def args_dgz(parser, remaining_args,args):
    parser.add_argument('--dataroot', default='./data', help='path to dataset')
    parser.add_argument('--image_embedding', default='res101')
    parser.add_argument('--class_embedding', default='att')
    parser.add_argument('--syn_num', type=int, default=100, help='number samples to generate per class')
    parser.add_argument('--preprocessing', action='store_true', default=True, help='enbale MinMaxScaler on visual features')
    parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
    parser.add_argument('--attSize', type=int, default=512, help='size of attribute features')
    parser.add_argument('--nz0', type=int, default=512, help='size of the latent z vector')
    parser.add_argument('--nepoch_classifier', type=int, default=60, help='number of epochs to train for the mapping net')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
    parser.add_argument('--lambda_0', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
    # parser.add_argument('--lr', type=float, default=1e-4, help='learning rate to train GANs ')
    parser.add_argument('--lr_classifier', type=float, default=1e-4, help='learning rate to train mapping net')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--manualSeed', type=int, default=70, help='manual seed, default=1429')
    parser.add_argument('--netD_layer_sizes', type=list, default=[4096])
    parser.add_argument('--netG_layer_sizes', type=list, default=[4096,2048,2048])
    parser.add_argument('--netM_layer_sizes', type=list, default=[1024,2048])
    parser.add_argument('--att_std', type=float, default=0.08, help='std of the attribute augmentation noise')
    parser.add_argument('--temperature', type=float, default=0.04)
    parser.add_argument('--lambda_1', type=float, default=0.005)
    # parser.add_argument("--datadir", type=str,default='/workspace/arijit/sandipan/zsgr_caddy/hariansh/')
    args = parser.parse_args(remaining_args, namespace=args)
    return args

def args_haszsl(parser, remaining_args,args):
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--image_embedding', default='res101')
    parser.add_argument('--class_embedding', default='att')
    parser.add_argument('--preprocessing', action='store_true', default=True,
                        help='enbale MinMaxScaler on visual features')
    parser.add_argument('--standardization', action='store_true', default=False)
    parser.add_argument('--ol', action='store_true', default=False,
                        help='original learning, use unseen dataset when training classifier')
    parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
    parser.add_argument('--classifier_lr', type=float, default=1e-6, help='learning rate to train softmax classifier')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
    parser.add_argument('--pretrain_classifier', default='', help="path to pretrain classifier (to continue training)")
    parser.add_argument('--manualSeed', type=int, default=3131, help='manual seed 3483')
    parser.add_argument('--device', type=str, default="0")
    parser.add_argument('--matdataset', default=True, help='Data in matlab format')
    parser.add_argument('--size', type=int, default="448")
    parser.add_argument('--train_id', type=int, default=0)
    parser.add_argument('--pretrained', default=None, help="path to pretrain classifier (to continue training)")
    parser.add_argument('--pretrain_epoch', type=int, default=5)
    parser.add_argument('--pretrain_lr', type=float, default=1e-4, help='learning rate to pretrain model')
    parser.add_argument('--all', action='store_true', default=False)
    parser.add_argument("--datadir", type=str,default='/workspace/arijit/sandipan/zsgr_caddy/hariansh/')

    parser.add_argument('--gzsl', action='store_true')
    parser.add_argument('--additional_loss', action='store_true', default=True)

    parser.add_argument('--xe', type=float, default=1)
    parser.add_argument('--attri', type=float, default=1e-2)
    parser.add_argument('--l_xe', type=float, default=1)
    parser.add_argument('--l_attri', type=float, default=5e-2)

    parser.add_argument('--calibrated_stacking', type=float, default=0.5,
                        help='calibrated_stacking, shrinking the output score of seen classes')

    parser.add_argument('--avg_pool', action='store_true')

    parser.add_argument('--only_evaluate', action='store_true', default=False)
    parser.add_argument('--resume', default="")

    parser.add_argument('--perturb_lr', default=3, type=float)
    parser.add_argument('--loops_adv', default=30, type=int)
    parser.add_argument('--entropy_cls', default=10, type=float)
    parser.add_argument('--entropy_attention', default=-3, type=float)
    parser.add_argument('--latent_weight', default=1, type=float)
    parser.add_argument('--sim_weight', default=30, type=float)
    parser.add_argument('--zsl_weight', default=1, type=float)
    parser.add_argument('--attention_sup', default=0.1, type=float)
    parser.add_argument('--perturb_start_epoch', default=0, type=int)
    parser.add_argument('--prob_perturb', default=0.5, type=float)
    parser.add_argument('--weight_perturb', default=8.0, type=float)
    parser.add_argument('--resnet_path',type=str)

    args = parser.parse_args(remaining_args, namespace=args)

    args.dataroot = args.root + 'data'

    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])

    args.max_val = np.array([(1. - mean[0]) / std[0],
                        (1. - mean[1]) / std[1],
                        (1. - mean[2]) / std[2],
                        ])

    args.min_val = np.array([(0. - mean[0]) / std[0],
                        (0. - mean[1]) / std[1],
                        (0. - mean[2]) / std[2],
                        ])

    args.eps_size = np.array([abs((1. - mean[0]) / std[0]) + abs((0. - mean[0]) / std[0]),
                        abs((1. - mean[1]) / std[1]) + abs((0. - mean[1]) / std[1]),
                        abs((1. - mean[2]) / std[2]) + abs((0. - mean[2]) / std[2]),
                        ])

    args.eps = args.weight_perturb/255.


    # define attribute groups
    # if args.dataset == 'CUB':
    #     args.parts = ['head', 'belly', 'breast', 'belly', 'wing', 'tail', 'leg', 'others']
    #     args.group_dic = json.load(open(os.path.join(args.root, 'data', args.dataset, 'attri_groups_8.json')))
    #     args.sub_group_dic = json.load(open(os.path.join(args.root, 'data', args.dataset, 'attri_groups_8_layer.json')))
    #     args.resnet_path = '../pretrained_models/resnet101_c.pth.tar'
    # elif args.dataset == 'AWA2':
    #     args.parts = ['color', 'texture', 'shape', 'body_parts', 'behaviour', 'nutrition', 'activativity', 'habitat',
    #             'character']
    #     args.group_dic = json.load(open(os.path.join(args.root, 'data', args.dataset, 'attri_groups_9.json')))
    #     args.sub_group_dic = {}
    #     args.resnet_path = '../pretrained_models/resnet101-5d3b4d8f.pth'
    # elif args.dataset == 'SUN':
    #     args.parts = ['functions', 'materials', 'surface_properties', 'spatial_envelope']
    #     args.group_dic = json.load(open(os.path.join(args.root, 'data', args.dataset, 'attri_groups_4.json')))
    #     args.sub_group_dic = {}
    #     args.resnet_path = '../pretrained_models/resnet101_sun.pth.tar'        opt.resnet_path = './pretrained_models/resnet101-5d3b4d8f.pth'

    args.resnet_path = '/workspace/arijit/sandipan/zsgr_caddy/hariansh/ZSL_models/HASZSL/pretrained_models/resnet101-5d3b4d8f.pth'

    args.reg_weight = {'final': {'xe': args.xe, 'attri': args.attri},
                'layer4': {'l_xe': args.l_xe, 'attri': args.l_attri},  # l denotes layer
                }
    
    return args

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--root', default='/workspace/arijit/sandipan/zsgr_caddy/hariansh', type=str, help='Root directory')
    parser.add_argument('--image_folder', required=True, type=str)
    parser.add_argument('--enhancement_path', default='', type=str)
    parser.add_argument('--dataset', default='CADDY', type=str)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--split_path', default='splits', type=str)
    parser.add_argument('--split_type', default='random', type=str)
    parser.add_argument('--split_number', default=1, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--num_classes', default=16, type=int)
    parser.add_argument('--use_enhanced_imgs', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--clip_version', default='ViT-B/32', type=str)
    parser.add_argument('--output_dir', default='output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--method', default='ours', help='ours, clip_linear_probe, pretrained_cnn, existing_zsl')
    parser.add_argument('--our_method_type', default='base-ViT', help='base-ViT')
    parser.add_argument('--pretrained_cnn_type', default='resnet50', help='resnet50, vgg16, resnet101')
    parser.add_argument('--existing_zsl_type', default='tfvaegan', help='tfvaegan')
    parser.add_argument('--setting', default='final', type=str)
    parser.add_argument('--use_last_chk_point', action='store_true')
    parser.add_argument('--dataroot', default='data', help='path to dataset')
    parser.add_argument('--image_embedding', default='res101')
    parser.add_argument('--class_embedding', default='att')
    parser.add_argument('--syn_num', type=int, default=300, help='number features to generate per class')
    parser.add_argument('--gzsl', action='store_true', default=True, help='enable generalized zero-shot learning')
    parser.add_argument('--preprocessing', action='store_true', help='enbale MinMaxScaler on visual features')
    parser.add_argument('--preprocessing_clip', action='store_true', help='enbale MinMaxScaler on visual features')
    parser.add_argument('--standardization', action='store_true', default=False)
    parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
    parser.add_argument('--attSize', type=int, default=512, help='size of semantic features')
    parser.add_argument('--nz', type=int, default=512, help='size of the latent z vector')
    parser.add_argument('--nz0', type=int, default=512, help='size of the latent z vector')
    parser.add_argument('--nepoch_classifier', type=int, default=50, help='number of epochs to train for the mapping net')
    parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
    parser.add_argument('--ndh', type=int, default=4096, help='size of the hidden units in discriminator')
    parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
    parser.add_argument('--cls_nepoch', type=int, default=2000, help='number of epochs to train for classifier')
    parser.add_argument('--lambda_0', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
    parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
    parser.add_argument('--lambda2', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs ')
    parser.add_argument('--feed_lr', type=float, default=0.00001, help='learning rate to train GANs ')
    parser.add_argument('--dec_lr', type=float, default=0.0001, help='learning rate to train GANs ')
    parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
    parser.add_argument('--lr_classifier', type=float, default=1e-4, help='learning rate to train mapping net')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
    parser.add_argument('--encoded_noise', action='store_true', default=True, help='enables validation mode')
    parser.add_argument('--manualSeed', type=int,default=3483, help='manual seed')
    parser.add_argument('--validation', action='store_true', default=False, help='enables validation mode')
    parser.add_argument("--encoder_layer_sizes", type=list, default=[8192, 4096])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[4096, 8192])
    parser.add_argument('--gammaD', type=int, default=10, help='weight on the W-GAN loss')
    parser.add_argument('--gammaG', type=int, default=10, help='weight on the W-GAN loss')
    parser.add_argument('--gammaG_D2', type=int, default=1000, help='weight on the W-GAN loss')
    parser.add_argument('--gammaD2', type=int, default=1000, help='weight on the W-GAN loss')
    parser.add_argument("--latent_size", type=int, default=512)
    parser.add_argument("--conditional", action='store_true',default=True)
    parser.add_argument("--datadir", type=str,default='/workspace/arijit/sandipan/zsgr_caddy/hariansh/')
    ###
    parser.add_argument('--a1', type=float, default=1.0)
    parser.add_argument('--a2', type=float, default=1.0)
    parser.add_argument('--recons_weight', type=float, default=0.01, help='recons_weight for decoder')
    parser.add_argument('--feedback_loop', type=int, default=2)
    parser.add_argument('--freeze_dec', action='store_true', default=False, help='Freeze Decoder for fake samples')
    
    parser.add_argument('--result_root', type=str, default='/workspace/arijit/sandipan/zsgr_caddy/hariansh/ZSL_models/FREE', help='root path for saving checkpoint')
    parser.add_argument('--center_margin', type=float, default=200, help='the margin in the center loss')
    parser.add_argument('--center_weight', type=float, default=0.5, help='the weight for the center loss')
    parser.add_argument('--incenter_weight', type=float, default=0.8, help='the weight for the center loss')
    parser.add_argument('--cls_weight', type=float, default=0.2, help='weight of the classification loss')
    parser.add_argument('--nclass_seen', type=int, default=10, help='number of seen classes')
    parser.add_argument('--latensize', type=int, default=2048, help='size of semantic features')
    parser.add_argument('--i_c', type=float, default=0.1, help='information constrain')
    parser.add_argument('--lr_dec', action='store_true', default=False, help='enable lr decay or not')
    parser.add_argument('--lr_dec_ep', type=int, default=1, help='lr decay for every 100 epoch')

    parser.add_argument('--netD_layer_sizes', type=list, default=[4096])
    parser.add_argument('--netG_layer_sizes', type=list, default=[4096,2048,2048])
    parser.add_argument('--netM_layer_sizes', type=list, default=[1024,2048])
    parser.add_argument('--att_std', type=float, default=0.08, help='std of the attribute augmentation noise')
    parser.add_argument('--temperature', type=float, default=0.04)
    parser.add_argument('--lambda_1', type=float, default=0.005)
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")

    parser.add_argument('--ol', action='store_true', default=False,
                        help='original learning, use unseen dataset when training classifier')
    parser.add_argument('--pretrain_classifier', default='', help="path to pretrain classifier (to continue training)")
    parser.add_argument('--device', type=str, default="0")
    parser.add_argument('--size', type=int, default="448")

    args = parser.parse_args()
    args.lambda2 = args.lambda1
    args.encoder_layer_sizes[0] = args.resSize
    args.decoder_layer_sizes[-1] = args.resSize
    args.latent_size = args.attSize

    # parser.add_argument('--num_cross_attention_layers', default=3, action='store_true')
    # parser.add_argument('--wandb_enable', default=False, action='store_true')
    # parser.add_argument('--concat_v_fo', action='store_true')
    # parser.add_argument('--gated_cross_attn', action='store_true')
    # parser.add_argument('--rho_eps', default=1e-14, type=float)
    # parser.add_argument('--hc_eps', default=1e-7, type=float)
    # parser.add_argument('--kappa', default=2.0, type=float)
    # parser.add_argument('--conceptnet_path', default="",type=str)
    # parser.add_argument('--proposed_loss', action='store_true')
    # parser.add_argument('--extend_lr_drop_list', action='store_true')
    # parser.add_argument('--rel_alpha', default=0.5, type=float)
    print(args)
    return args

def main(args):
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # _, clip_preprocess = clip.load('ViT-B/32', device)
    # exit(0)
    print('setting up seeds')
    setup_seed(args.seed)

    dirs = setup_dirs(args)
    args.dirs = dirs
    if args.method == "existing_zsl" and args.existing_zsl_type == "transzero_pp":
        train_cub.train_transzero_pp(args)
        return
    elif args.method == "existing_zsl" and args.existing_zsl_type == "tfvaegan":
        train_images.train_tfvaegan(args)
        return
    elif args.method == "existing_zsl" and args.existing_zsl_type == "FREE":
        train_free.train_FREE(args)
        return
    elif args.method == "existing_zsl" and args.existing_zsl_type == "DGZ":
        train_DGZ.train_dgz(args)
        return

    # split naming conventions:
    # [X]_[Y]_[num].csv =>, where:
    # [X] = seen or unseen
    # [Y] = random or RF or NF
    # [num] = 1, 2, 3.... if X = random else 0

    train_split_name = 'train_' + args.split_type + '_' + str(args.split_number) 
    test_seen_split_name = 'test_seen_' + args.split_type + '_' + str(args.split_number) 
    test_unseen_split_name = 'test_unseen_' + args.split_type + '_' + str(args.split_number)
    train_path = args.dirs['split'] + '/' + train_split_name + '.csv' # change
    test_seen_path = args.dirs['split'] + '/' + test_seen_split_name + '.csv' # change
    test_unseen_path = args.dirs['split']  + '/' + test_unseen_split_name + '.csv'
    if Path(train_path).is_file() and Path(test_seen_path).is_file() and Path(test_unseen_path).is_file():
        print('Split files found!')
        args.train_path, args.test_seen_path, args.test_unseen_path = train_path, test_seen_path, test_unseen_path    # add another flag for train
    else:
        print('Split files not found! Creating new splits...')
        if args.dataset == 'CADDY':
            args.train_path, args.test_seen_path, args.test_unseen_path = make_caddy_splits(train_path, test_seen_path, test_unseen_path, args)
            return
        else:
            raise ValueError(f'dataset {args.dataset} not supported')

    print('Initializing dataloaders...')

    args.split_labels = get_split_labels(args.train_path, args.test_seen_path, args.test_unseen_path)

    if args.method == "ours" and args.our_method_type == "GAN":
        method_runner_gan.our_method(args)
        return

    training_dataset = build_dataset('train', args)
    print(f'Total training images: {len(training_dataset)}')

    if args.distributed:
        sampler_train = DistributedSampler(training_dataset)
    else:
        sampler_train = torch.utils.data.RandomSampler(training_dataset)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(training_dataset, batch_sampler=batch_sampler_train, num_workers=args.num_workers, pin_memory=True)

    test_seen_dataset = build_dataset(image_set='test_seen', args=args)
    print(f'Total test seen images: {len(test_seen_dataset)}')
    
    test_unseen_dataset = build_dataset(image_set='test_unseen', args=args)
    print(f'Total test unseen images: {len(test_unseen_dataset)}')
    if args.distributed:
        sampler_test_seen = DistributedSampler(test_seen_dataset, shuffle=False)
        sampler_test_unseen = DistributedSampler(test_unseen_dataset, shuffle=False)
    else:
        sampler_test_seen = torch.utils.data.SequentialSampler(test_seen_dataset)
        sampler_test_unseen = torch.utils.data.SequentialSampler(test_unseen_dataset)

    data_loader_test_seen = DataLoader(test_seen_dataset, args.batch_size, sampler=sampler_test_seen,
                                 drop_last=False, num_workers=args.num_workers, pin_memory=True)
    data_loader_test_unseen = DataLoader(test_unseen_dataset, args.batch_size, sampler=sampler_test_unseen,
                                  drop_last=False, num_workers=args.num_workers, pin_memory=True)
    print('Dataloaders finished!')

    if args.method == "existing_zsl" and args.existing_zsl_type == "HASZSL":
        train_HASZSL.train_haszsl(args,data_loader_train,data_loader_test_unseen,data_loader_test_seen)
        return

    datasets= {'train':training_dataset, 'test_seen':test_seen_dataset, 'test_unseen': test_unseen_dataset}
    dataloaders= {'train':data_loader_train, 'test_seen': data_loader_test_seen, 'test_unseen': data_loader_test_unseen}

    if args.method == "pretrained_cnn":
        method_runner_cnn.supervised_method(datasets, dataloaders, args)
        return

    if args.method == "ours" and args.our_method_type == "GCAT":
        method_runner.our_method(datasets, dataloaders, args)
        return
    elif args.method == "clip_linear_probe":
        print("Clip linear probe in main.py")
        method_runner.our_method(datasets, dataloaders, args)
        return
    elif args.method == "preprocessing" or args.method == "preprocessing_clip":
        extract_features(args, dataloaders)
        return
    else:
        print("select a correct method\n")

if __name__ == '__main__':
    args, remaining_args, parser = get_args_parser_known()
    if args.method == "ours" and args.our_method_type == "GAN":
        args = args_GAN(parser, remaining_args, args)
    elif args.method == "existing_zsl":
        if args.existing_zsl_type == "tfvaegan":
            args = args_tfvaegan(parser, remaining_args, args)
        elif args.existing_zsl_type == "FREE":
            args = args_free(parser, remaining_args, args)
        elif args.existing_zsl_type == "DGZ":
            args = args_dgz(parser, remaining_args, args)
        elif args.existing_zsl_type == "HASZSL":
            args = args_haszsl(parser, remaining_args, args)  

    main(args)
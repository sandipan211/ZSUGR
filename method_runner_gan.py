import torch
import torch.nn as nn
import os
import time
from torch.optim import Adam
from VisionTransformer.custom import VisionTransformer, CLIPClassifier
from VisionTransformer.logger import create_logger
import clip
from models import network as gcat_network
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler.cosine_lr import CosineLRScheduler
import util.trainval as TV
from util import train_gan
import h5py
import numpy as np
from models import classifier
from models import pretrained_classifier

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

def get_data(args):
    args.datadir = os.path.join(args.datadir, 'data/{}/'.format(args.dataset))
    path = os.path.join(args.datadir, '{}_{}'.format(args.split_type,args.split_number))
    if args.use_resnet:
        path = path = os.path.join(path, f'feature_map_ResNet_101_CADDY_2048.hdf5')
    else:
        path = os.path.join(path, f'gcat_features_{args.setting}.hdf5')

    print('_____')
    print(path)

    hf = h5py.File(path, 'r')
    train_feature = np.array(hf.get('feature_map_train')) 
    test_seen_feature = np.array(hf.get('feature_map_test_seen')) 
    test_unseen_feature = np.array(hf.get('feature_map_test_unseen'))

    train_label = np.array(hf.get('labels_train'))
    test_seen_label = np.array(hf.get('labels_test_seen'))
    test_unseen_label = np.array(hf.get('labels_test_unseen'))

    train_feature = torch.from_numpy(train_feature).float()
    train_label = torch.from_numpy(train_label).long() 
    test_unseen_feature = torch.from_numpy(test_unseen_feature).float()
    test_unseen_label = torch.from_numpy(test_unseen_label).long() 
    test_seen_feature = torch.from_numpy(test_seen_feature).float() 
    test_seen_label = torch.from_numpy(test_seen_label).long()

    return train_feature, train_label, test_unseen_feature, test_unseen_label, test_seen_feature, test_seen_label

def our_method(args):

    train_feature, train_label, test_unseen_feature, test_unseen_label, test_seen_feature, test_seen_label = get_data(args)
    # print(test_seen_feature.size())
    # print(test_seen_label.size())
    # print(test_unseen_feature.size())
    # print(test_unseen_label.size())
    # print(train_feature.size())
    # print(train_label.size())
    data = {'test_seen_feature': test_seen_feature, 'test_seen_label':test_seen_label, 'test_unseen_feature': test_unseen_feature, 'test_unseen_label': test_unseen_label, 'unseen_classes': args.split_labels['test_unseen'], 'seen_classes': args.split_labels['train']}
    if args.method == 'ours':
        dir = args.dirs[f'{args.our_method_type}_{args.split_type}_{args.split_number}']
    elif args.method == 'clip_linear_probe':
        dir = args.dirs[f'{args.clip_version}_{args.split_type}_{args.split_number}']
    elif args.method == 'pretrained_cnn':
        dir = args.dirs[f'{args.pretrained_cnn_type}_{args.split_type}_{args.split_number}']
    elif args.method == 'existing_zsl':
        dir = args.dirs[f'{args.existing_zsl_type}_{args.split_type}_{args.split_number}']
    else:
        raise ValueError(f'Method {args.method} not supported')

    output_file_path = os.path.join(dir,'log_train_'+args.setting+'.txt')
    f = open(output_file_path,'a')
    f.write(str(args))
    sentence_features = get_sentence_embeddings()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainGAN = train_gan.TrainGAN(args, sentence_features, f)
    since = time.time()
    args.unseen_labels = args.split_labels['test_unseen']
    num_epochs = args.epochs # take from args

    ckpt_path = os.path.join(dir, args.setting+'_'+args.resume)
    best_ckpt_path = os.path.join(dir, args.setting+'_'+args.best_ckpt)
    if args.resume and os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        print(f'Loaded checkpoint from {ckpt_path}')
        trainGAN.netG.load_state_dict(checkpoint['netG'])
        trainGAN.netD.load_state_dict(checkpoint['netD'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        # print("start 0\n")
        start_epoch = 0

    if args.continue_lastbest_performance and os.path.exists(best_ckpt_path):
        best_checkpoint = torch.load(best_ckpt_path, map_location='cpu')
        best_performance = best_checkpoint['performance_stats_and_classifers']
    else:
        # TODO: make best scores for acc_seen, acc_unseen, acc_H, too
        # best_performance = {
        #     'acc_novel': 0.0,
        #     'acc_seen': 0.0,
        #     'acc_unseen': 0.0,
        #     'HM': 0.0
        # }
        # print("new bp\n")

        best_performance = {
            'best_czsl_acc': 0.0,
            'best_acc_seen': 0.0,
            'best_acc_unseen': 0.0,
            'best_H': 0.0,
            'best_gzsl_model': {
                'netG': {},
                'netD': {},
                'classifier': {}
            },
            'best_acc_seen_per_class_gzsl': {},
            'best_acc_unseen_per_class_gzsl': {},
            'best_czsl_model': {
                'netG': {},
                'netD': {},
                'classifier': {}
            },
            'best_acc_per_class_czsl': {},
            'args': args
        }

    best_gzsl_acc = best_performance['best_H']
    best_zsl_acc = best_performance['best_czsl_acc']
    best_acc_seen = best_performance['best_acc_seen']
    best_acc_unseen = best_performance['best_acc_unseen']
    best_gzsl_model = best_performance['best_gzsl_model']
    best_acc_seen_per_class = best_performance['best_acc_seen_per_class_gzsl']
    best_acc_unseen_per_class = best_performance['best_acc_unseen_per_class_gzsl']
    best_czsl_model = best_performance['best_czsl_model']
    best_acc_per_class_czsl = best_performance['best_acc_per_class_czsl']
    # pretrain_cls = pretrained_classifier.CLASSIFIER(train_feature, map_label(train_label,args.split_labels['train']), len(args.split_labels['train']), args.resSize, args.cuda, 0.001, 0.5, 50, 100, args.pretrain_classifier)
    # for p in pretrain_cls.model.parameters(): 
    #     p.requires_grad = False

    for epoch in range(start_epoch, num_epochs):
        # f.write('-' * 10)
        # Training GAN
        # trainGAN(epoch, train_feature, train_label , device, f, pretrain_cls)
        trainGAN(epoch, train_feature, train_label , device, f)
        # generate features from GAN and train classifier
        syn_features, syn_labels = trainGAN.generate_syn_feature(args.unseen_labels, sentence_features[args.unseen_labels],num=args.syn_num)
        if args.gzsl:   
            # Concatenate real seen features with synthesized unseen features
            train_X = torch.cat((train_feature, syn_features), 0)
            train_Y = torch.cat((train_label, syn_labels), 0)
            nclass = args.num_classes

            # Train GZSL classifier
            gzsl_cls = classifier.CLASSIFIER(train_X, train_Y, data, nclass, args.cuda, args.classifier_lr, 0.5, \
                    25, args.syn_num, generalized=True)
            if best_gzsl_acc < gzsl_cls.H:
                best_acc_seen, best_acc_unseen, best_gzsl_acc, best_gzsl_classifier, best_acc_seen_per_class, best_acc_unseen_per_class \
                    = gzsl_cls.acc_seen, gzsl_cls.acc_unseen, gzsl_cls.H, gzsl_cls.best_model_gzsl, gzsl_cls.acc_seen_per_class, gzsl_cls.acc_unseen_per_class
                best_gzsl_model = {
                    'netG': trainGAN.netG.state_dict(),
                    'netD': trainGAN.netD.state_dict(),
                    'classifier': best_gzsl_classifier
                }
            f.write('Epoch: %d GZSL: seen=%.4f, unseen=%.4f, h=%.4f \n' % (epoch, gzsl_cls.acc_seen, gzsl_cls.acc_unseen, gzsl_cls.H))
            print('Epoch: %d GZSL: seen=%.4f, unseen=%.4f, h=%.4f \n' % (epoch, gzsl_cls.acc_seen, gzsl_cls.acc_unseen, gzsl_cls.H))

        # Zero-shot learning
        # Train ZSL classifier
        zsl_cls = classifier.CLASSIFIER(syn_features, map_label(syn_labels, data['unseen_classes']), \
                        data, len(data['unseen_classes']), args.cuda, args.classifier_lr, 0.5, 25, args.syn_num, \
                        generalized=False)

        acc = zsl_cls.acc
        if best_zsl_acc < acc:
            best_zsl_acc, best_czsl_classifier, best_acc_per_class_czsl = zsl_cls.acc, zsl_cls.best_model_czsl, zsl_cls.acc_per_class_czsl
            best_czsl_model = {
                'netG': trainGAN.netG.state_dict(),
                'netD': trainGAN.netD.state_dict(),
                'classifier': best_czsl_classifier
            }
        f.write('ZSL: unseen accuracy=%.4f \n' % (acc)) 
        print('ZSL: unseen accuracy=%.4f \n' % (acc)) 
      
        torch.save({
            'netG': trainGAN.netG.state_dict(), 
            'netD': trainGAN.netD.state_dict(),
            'epoch': epoch, 
            'performance_stats': {
                'best_czsl_acc': best_zsl_acc,
                'best_acc_seen': best_acc_seen,
                'best_acc_unseen': best_acc_unseen,
                'best_H': best_gzsl_acc
            },
            'args': args           
        }, ckpt_path)

    f.write(f'Dataset {args.dataset} \n')
    f.write(f'the best CZSL unseen accuracy is {best_zsl_acc} \n')
    f.write(f'Class-wise acc CZSL: {best_acc_per_class_czsl}\n')
    if args.gzsl:
        f.write(f'Dataset {args.dataset}')
        f.write(f'the best GZSL seen accuracy is {best_acc_seen} \n')
        f.write(f'the best GZSL unseen accuracy is {best_acc_unseen} \n')
        f.write(f'the best GZSL H is {best_gzsl_acc} \n')
        f.write(f'Class-wise acc seen GZSL: {best_acc_seen_per_class}\n')
        f.write(f'Class-wise acc unseen GZSL: {best_acc_unseen_per_class}\n')
        print(f'Dataset {args.dataset}')
        print(f'the best GZSL seen accuracy is {best_acc_seen} \n')
        print(f'the best GZSL unseen accuracy is {best_acc_unseen} \n')
        print(f'the best GZSL H is {best_gzsl_acc} \n')

    time_elapsed = time.time() - since
    f.write(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s \n")
    f.close()
    torch.save({'performance_stats_and_classifiers': {
                                                        'best_gzsl_model': best_gzsl_model,
                                                        'best_H': best_gzsl_acc,
                                                        'best_acc_seen': best_acc_seen,
                                                        'best_acc_unseen': best_acc_unseen,
                                                        'best_acc_seen_per_class_gzsl': best_acc_seen_per_class,
                                                        'best_acc_unseen_per_class_gzsl': best_acc_unseen_per_class,
                                                        'best_czsl_model': best_czsl_model,
                                                        'best_czsl_acc': best_zsl_acc,
                                                        'best_acc_per_class_czsl': best_acc_per_class_czsl,
                                                        'args': args  
                                                    }         
        }, best_ckpt_path)
#-----------

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(len(classes)):
        mapped_label[label==classes[i]] = i    

    return mapped_label
import torch
import torch.nn as nn
import os
import time
from torch.optim import Adam
from VisionTransformer.custom import VisionTransformer, CLIPClassifier
# from VisionTransformer.logger import create_logger
import clip
from models import network
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler.cosine_lr import CosineLRScheduler
import util.trainval as TV
from util import train_gan


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

def load_ckpt(args, ckpt_path, optimizer, lr_scheduler):
        
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        # change SS: do this part if you want to add more lr drops while your code has already run for k times
        # if args.extend_lr_drop_list:
        #     lr_drops = args.lr_drop_list
        #     # make sure you have added the lr drop epochs in the script to run
        #     lr_scheduler.milestones = Counter(lr_drops)
        #     # if lr scheduler has to make a drop at epoch k, it checks for it at epoch k-1 
        #     # print(lr_scheduler.milestones)
    return checkpoint

def our_method(datasets,dataloaders,args):

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

    # logger = create_logger(output_dir=dir, eval=args.eval, setting_name=args.setting) # take from arguments
    output_file_path = os.path.join(dir,'log_train_'+args.setting+'.txt')
    f = open(output_file_path,'a')
    f.write(str(args))

    if args.our_method_type == 'base-ViT':
        f.write('\n = = = = = = = = Vision Transformer = = = = = = = = \n' )
    elif args.our_method_type == 'GCAT':
        f.write('\n = = = = = = = = GCAT model = = = = = = = = \n' )

    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'test_seen', 'test_unseen']}
    sentence_features = get_sentence_embeddings()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(f'device: {device}')
    # device=torch.device("cpu")
    num_epochs = args.epochs # take from args

    if args.method == 'ours':
        print('###########################')
        print(args.our_method_type)
        print('###########################')
        if args.our_method_type == 'base-ViT':
            model = VisionTransformer(
                img_width=224,  
                img_height=224,
                patch_size=16, # subject to change
                in_chans=3,
                n_classes=16,  # Change this based on your dataset
                embed_dim=768,
                depth=12,
                n_heads=12,
                mlp_ratio=4.,
                qkv_bias=True,
                p=0.,
                attn_p=0.,
            ).to(device)
            
            # Define optimizer and scheduler
            optimizer = Adam(model.parameters(), lr=2e-6)
            n_iter_per_epoch = len(dataloaders['train'])
            num_steps = int(num_epochs * n_iter_per_epoch)
            warmup_steps = int(5 * n_iter_per_epoch)
            lr_scheduler = CosineLRScheduler(
                optimizer,
                t_initial=num_steps,
                lr_min=8.e-6 / 100,
                warmup_lr_init=0,
                warmup_t=warmup_steps,
                cycle_limit=1,
                t_in_epochs=False,
            )   
        elif args.our_method_type == 'GCAT':
            model = network.build(args, sentence_features).to(device)
            # no need to compute gradients through this linear layer during test time
            for name, p in model.named_parameters():
                if 'eval_visual_projection' in name:
                    p.requires_grad = False

            # no need to compute gradients for CLIP components
            if args.fix_clip:
                for name, p in model.named_parameters():
                    if 'visual_projection' in name or 'clip_model' in name:
                        p.requires_grad = False

            param_dicts = [
            {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": args.lr_backbone,
            },
            ]

            optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print('Number of params:', n_parameters)

            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.lr_drop], gamma=args.lr_drop_gamma)

    elif args.method == 'clip_linear_probe':
        model = CLIPClassifier(args, 512).to(device)

    ckpt_path = os.path.join(dir, args.setting+'_'+args.resume)
    best_ckpt_path = os.path.join(dir, args.setting+'_'+args.best_ckpt)
    if args.resume and os.path.exists(ckpt_path):
        checkpoint = load_ckpt(args, ckpt_path, optimizer, lr_scheduler)
        print(f'Loaded checkpoint from {ckpt_path}')
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0

    if os.path.exists(best_ckpt_path):
        best_checkpoint = load_ckpt(args, best_ckpt_path, optimizer, lr_scheduler)
        best_performance = best_checkpoint['performance_stats']
    else:
        # TODO: make best scores for acc_seen, acc_unseen, acc_H, too
        # best_performance = {
        #     'acc_novel': 0.0,
        #     'acc_seen': 0.0,
        #     'acc_unseen': 0.0,
        #     'HM': 0.0
        # }
        best_performance = {
            'acc_seen': 0.0,
            'acc_unseen': 0.0,
            'HM': 0.0
        }


    since = time.time()
    args.unseen_labels = args.split_labels['test_unseen']
    # best_acc = 0.0
    # best_H = 0.0 
    # best_seen_acc = 0.0
    # best_unseen_acc = 0.0
    for epoch in range(start_epoch, num_epochs):
        epoch_since = time.time()
        # f.write('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'test_unseen']:
            if phase == 'train':
                # return avg loss
                avg_loss = TV.train_one_epoch(epoch,device, model, dataloaders[phase],optimizer,lr_scheduler,f,sentence_features, args)  # Set model to training mode

            else:
                # acc1_avg, _ = TV.val_czsl(device, dataloaders[phase], model)   # Set model to evaluate mode
                H, acc_seen, _, acc_unseen, _ = TV.val_gzsl(device, dataloaders, model)
                epoch_time = time.time() - epoch_since
                f.write(f'Epoch {epoch+1}/{num_epochs} ')
                # f.write(f'acc_czsl = {acc1_avg*100} ')
                f.write(f'loss={avg_loss} H={H*100} acc_gzsl: acc_seen={acc_seen*100} acc_unseen={acc_unseen*100}\n')
                print(f'Epoch [{epoch+1}/{num_epochs}] loss: {avg_loss} H: {H*100: .6f} acc_seen: {acc_seen*100: .6f} acc_unseen: {acc_unseen*100: .6f} time: {epoch_time // 60:.0f}m {epoch_time % 60:.0f}s')
                # print(f'Epoch [{epoch+1}/{num_epochs}] loss: {avg_loss} acc_czsl: {acc1_avg*100: .6f} H: {H*100: .6f} acc_seen: {acc_seen*100: .6f} acc_unseen: {acc_unseen*100: .6f} time: {epoch_time // 60:.0f}m {epoch_time % 60:.0f}s')
            
            lr_scheduler.step()

            if phase == 'test_unseen' and H > best_performance['HM']:
                best_performance['acc_seen']=acc_seen
                best_performance['acc_unseen']=acc_unseen
                best_performance['HM'] = H
                deep_copy = {'avg_loss': avg_loss, 
                             'model': model.state_dict(), 
                             'optimizer': optimizer.state_dict(),
                             'lr_scheduler': lr_scheduler.state_dict(),
                             'epoch': epoch, 
                             'performance_stats': best_performance,
                             'args': args
                             }
                torch.save(deep_copy, best_ckpt_path)

            
            # save current epoch ckpt
            if phase == 'test_unseen':
                torch.save({
                    'avg_loss': avg_loss,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch, 
                    # 'performance_stats': {
                    #     'acc_novel': acc1_avg,
                    #     'acc_seen': acc_seen,
                    #     'acc_unseen': acc_unseen,
                    #     'HM': H

                    # },
                    'performance_stats': {
                        'acc_seen': acc_seen,
                        'acc_unseen': acc_unseen,
                        'HM': H
                    },
                    'args': args           
                }, ckpt_path)

    time_elapsed = time.time() - since
    print(f"\nBest GZSL: H={best_performance['HM']*100} acc_seen={best_performance['acc_seen']*100} acc_unseen={best_performance['acc_unseen']*100}")
    # print(f"Best CZSL: acc_czsl={best_performance['acc_novel']*100}")
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

    f.write(f"\n\nBest GZSL: H={best_performance['HM']*100} acc_seen={best_performance['acc_seen']*100} acc_unseen={best_performance['acc_unseen']*100}")
    # f.write(f"\nBest CZSL: acc_czsl={best_performance['acc_novel']*100}")
    f.write(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    f.close()
#-----------
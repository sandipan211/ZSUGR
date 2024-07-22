import torch
import torch.nn as nn
import os
import time
from torch.optim import Adam
import clip
from pretrained_models import resnet_101_zsl as resnet101


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


def supervised_method(datasets, dataloaders, args):
    dir = args.dirs[f'{args.pretrained_cnn_type}_{args.split_type}_{args.split_number}']
    output_file_path = os.path.join(dir,'log_train_'+args.setting+'.txt')
    f = open(output_file_path,'a')
    f.write(str(args))
    

    sentence_features = get_sentence_embeddings()

    if args.pretrained_cnn_type == 'resnet101':
        f.write('\n = = = = = = = = ResNet-101 = = = = = = = = \n' )
        f.close()
        resnet101.resnet_101_zsl(datasets, dataloaders, args, sentence_features, output_file_path, dir)


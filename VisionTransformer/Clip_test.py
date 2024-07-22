import sys

import torch
import clip
from PIL import Image
import numpy as np
from VisionTransformer.tools import AverageMeter
from VisionTransformer.logger import create_logger

sys.path.append('..')
from method_runner import get_sentence_embeddings

def validate_clip(val_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    logger = create_logger('/workspace/arijit/sandipan/zsgr_caddy/hariansh/VisionTransformer/output') # take from arguments
    
    sentence_features = get_sentence_embeddings()
    acc1_meter, acc5_meter = AverageMeter(), AverageMeter()
    true_label_set = set()
    predict_label_set = set()
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = preprocess(inputs).unsqueeze(0).to(device)
            inputs = inputs.to(device)
            labels = targets['label'].to(device)
            sentence_features = sentence_features.to(device)   

            b, _ , _ , _ = inputs.size()
            output = model.encode_image(inputs)
            
            _, preds = torch.max(output,1)

            values_1, indices_1 = output.topk(1, dim=-1)
            values_5, indices_5 = output.topk(5, dim=-1)
            acc1, acc5 = 0, 0
            for i in range(len(labels)):
                if indices_1[i] == labels[i]:
                    acc1 += 1
                if labels[i] in indices_5[i]:
                    acc5 += 1

            acc1_meter.update(float(acc1) / b * 100, b)
            acc5_meter.update(float(acc5) / b * 100, b)

            # print("type of preds : ",preds)
            # Convert tensor to a list of integers
            
            integer_list = [int(element.item()) for element in preds]
            integer_list2 = [int(element.item()) for element in labels]
            for ele in integer_list:
                predict_label_set.add(ele)
            for ele2 in integer_list2:
                true_label_set.add(ele2)
            # print("Predictions:", preds)
            # print("True Labels:", labels)

    logger.info("true_label_set: ")   
    logger.info(true_label_set)
    logger.info("predict_label_set: ")
    logger.info(predict_label_set)
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg
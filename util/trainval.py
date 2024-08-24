import models.losses as L
import torch
from VisionTransformer.tools import AverageMeter
import torch.nn.functional as F
import torch.nn as nn
import gc

# label id to label name mapping 
label_dict =    {
    # -1: "negative",
    0:  "start_comm",
    1:  "end_comm",
    2:  "up",
    3:  "down",
    4:  "photo",
    5:  "backwards",
    6:  "carry",
    7:  "boat",
    8:  "here",
    9:  "mosaic",
    10: "num_delimiter",
    11: "one",
    12: "two",
    13: "three",
    14: "four",
    15: "five"
}

def update_keys(acc_per_class):
    updated_acc_per_class = {label_dict.get(key, key): value for key, value in acc_per_class.items()}
    return updated_acc_per_class

def map_label(labels, classes):
    mapped_label = torch.LongTensor(labels.shape)
    for i in range(len(classes)):
        mapped_label[labels==classes[i]] = i    

    return mapped_label

def train_one_epoch(epoch,device,model,train_loader,optimizer,scheduler,f,sentence_features, args):
    
    model.train()
    
    tot_loss_meter = AverageMeter()
    num_steps = len(train_loader)
    # running_corrects = 0
    # Iterate over data.
    idx=0
    for inputs, targets in train_loader:
        # if idx%50 ==0 :
        #     f.write(f'Batch [{int(idx/50)}/{int(len(train_loader)/args.batch_size)}]')

        inputs = inputs.to(device)
        #label is returned as dictionary
        labels = targets['label']
        clip_input = targets['clip_inputs'].to(device)
        # print(f'clip input size = {clip_input.shape}')   -> [bs, 3, 224, 224]
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        # track history if only in train
        # outputs = model(inputs,sentence_features, args)
        if args.our_method_type == 'base-ViT':
            sentence_features = sentence_features.to(device)
            outputs = model(inputs,sentence_features,args)
            # _, preds = torch.max(outputs, 1)
            outputs = outputs.softmax(dim=-1) # additional
            loss = nn.CrossEntropyLoss(outputs, labels.to(device))
            # f.write(f'loss: {loss}')
            loss.backward()
            optimizer.step()
            scheduler.step_update(epoch * num_steps + idx)
            # statistics
            tot_loss_meter.update(loss.item(), inputs.size(0))
            # running_corrects += torch.sum(preds == labels.data)
            idx=idx+1

        elif args.our_method_type == 'GCAT':
            labels = map_label(labels, model.seen_labels).to(device)
            outputs = model(inputs, is_training=True, clip_input=clip_input)
            # total_loss = L.loss_gesture_labels(outputs, labels) + L.mimic_loss(outputs)
            total_loss = L.loss_gesture_labels(outputs, labels)
            total_loss.backward()
            optimizer.step()
            # statistics
            tot_loss_meter.update(total_loss.item(), inputs.size(0))
            # running_corrects += torch.sum(preds == labels.data)
            idx=idx+1

        inputs = None
        labels = None
        clip_input = None

    # f.write('[%d/%d]  tot_loss: %.4f '% (epoch, args.epochs, tot_loss_meter.avg))
    # f.write('\n')
    gc.collect()
    torch.cuda.empty_cache()
    return tot_loss_meter.avg

def val_czsl(device, val_loader, model):
    # unseen classes must be a list of unseen classes
    model.eval()
    acc, acc_per_class = val(val_loader, device, model, model.unseen_labels)
    acc_per_class = update_keys(acc_per_class)
    return acc, acc_per_class

def compute_per_class_acc(test_label, predicted_label, classes):
    acc_per_class = {}
    for i in classes:
        acc_per_class[i] = 0.0
    for i in classes:
        idx = (test_label == i)
        acc_per_class[i] = torch.sum(test_label[idx]==predicted_label[idx]) / torch.sum(idx)
    return acc_per_class

def val_gzsl(device, val_loader, model):
    # assuming val_loader to be dictionary having two dataloaders 'test_seen' and 'test_unseen'
    # unseen classes must be a list of unseen classes
    model.eval()
    
    acc_seen, acc_per_class_seen = val(val_loader['test_seen'], device, model, model.seen_labels)
    acc_unseen, acc_per_class_unseen = val(val_loader['test_unseen'], device, model, model.unseen_labels)
    H = 2*acc_seen*acc_unseen / (acc_seen+acc_unseen)
    acc_per_class_seen = update_keys(acc_per_class_seen)
    acc_per_class_unseen = update_keys(acc_per_class_unseen)
    # return mean (seen, unseen) accuracy
    # return HM
    # return classwise seen accs, classwise unseen accs (dicts)
    return H, acc_seen, acc_per_class_seen, acc_unseen, acc_per_class_unseen

def val(val_loader, device, model, classes):
    model.eval()
    predicted_labels_list = []
    true_labels_list = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            gt_labels = targets['label'].to(device)
            clip_input = targets['clip_inputs'].to(device)
            b, _ , _ , _ = inputs.size()
            outputs = model(inputs, is_training=False, clip_input=clip_input)
            # TODO: should we compute sigmoid? or will softmax do? Remember that we have not used signoid during loss calculation in training
            # gesture_scores = outputs['pred_gesture_logits'].sigmoid()
            # check the line below
            # pred_labels = F.softmax(outputs['pred_gesture_logits'], -1)[..., :-1].max(-1)[1]
            pred_labels = F.softmax(outputs['pred_gesture_logits'],-1)
            pred_labels = torch.argmax(pred_labels , dim=-1)
            predicted_labels_list.append(pred_labels)
            true_labels_list.append(gt_labels)

    predicted_labels = torch.cat(predicted_labels_list, dim=0)
    true_labels = torch.cat(true_labels_list, dim =0)
    acc_per_class = compute_per_class_acc(true_labels, predicted_labels, classes)
    acc = 0.0
    for key,value in acc_per_class.items():
        acc += value
    acc/= len(acc_per_class)

    return acc, acc_per_class

def val_gzsl_clip(device, val_loader, model):
    # assuming val_loader to be dictionary having two dataloaders 'test_seen' and 'test_unseen'
    # unseen classes must be a list of unseen classes
    model.eval()
    print("Evaluating clip")
    
    acc_seen, acc_per_class_seen = val_clip(val_loader['test_seen'], device, model, model.seen_labels)
    acc_unseen, acc_per_class_unseen = val_clip(val_loader['test_unseen'], device, model, model.unseen_labels)
    H = 2*acc_seen*acc_unseen / (acc_seen+acc_unseen)
    acc_per_class_seen = update_keys(acc_per_class_seen)
    acc_per_class_unseen = update_keys(acc_per_class_unseen)
    # return mean (seen, unseen) accuracy
    # return HM
    # return classwise seen accs, classwise unseen accs (dicts)
    return H, acc_seen, acc_per_class_seen, acc_unseen, acc_per_class_unseen

def val_clip(val_loader, device, model, classes):
    model.eval()
    predicted_labels_list = []
    true_labels_list = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            gt_labels = targets['label'].to(device)
            clip_input = targets['clip_inputs'].to(device)
            b, _ , _ , _ = inputs.size()
            outputs = model(inputs, is_training=False, clip_input=clip_input)
            # TODO: should we compute sigmoid? or will softmax do? Remember that we have not used signoid during loss calculation in training
            # gesture_scores = outputs['pred_gesture_logits'].sigmoid()
            # check the line below
            # pred_labels = F.softmax(outputs['pred_gesture_logits'], -1)[..., :-1].max(-1)[1]
            pred_labels = F.softmax(outputs['logits_per_image_all'],-1)
            pred_labels = torch.argmax(pred_labels , dim=-1)
            predicted_labels_list.append(pred_labels)
            true_labels_list.append(gt_labels)

    predicted_labels = torch.cat(predicted_labels_list, dim=0)
    true_labels = torch.cat(true_labels_list, dim =0)
    acc_per_class = compute_per_class_acc(true_labels, predicted_labels, classes)
    acc = 0.0
    for key,value in acc_per_class.items():
        acc += value
    acc/= len(acc_per_class)

    return acc, acc_per_class


# @torch.no_grad()
# def validate(device, val_loader, model, f, sentence_features, args):
#     model.eval()

#     acc1_meter, acc5_meter = AverageMeter(), AverageMeter()
#     true_label_set = set()
#     predict_label_set = set()
#     with torch.no_grad():
#         for inputs, targets in val_loader:
#             inputs = inputs.to(device)
#             labels = targets['label'].to(device)
#             sentence_features = sentence_features.to(device)   

#             b, _ , _ , _ = inputs.size()
#             # output = model(inputs,sentence_features, args)
#             output = model(inputs,sentence_features,args)
#             output = output.softmax(dim=-1)

#             _, preds = torch.max(output,1)

#             values_1, indices_1 = output.topk(1, dim=-1)
#             values_5, indices_5 = output.topk(5, dim=-1)
#             acc1, acc5 = 0, 0
#             for i in range(len(labels)):
#                 if indices_1[i] == labels[i]:
#                     acc1 += 1
#                 if labels[i] in indices_5[i]:
#                     acc5 += 1

#             acc1_meter.update(float(acc1) / b * 100, b)
#             acc5_meter.update(float(acc5) / b * 100, b)

#             # print("type of preds : ",preds)
#             # Convert tensor to a list of integers
            
#             integer_list = [int(element.item()) for element in preds]
#             integer_list2 = [int(element.item()) for element in labels]
#             for ele in integer_list:
#                 predict_label_set.add(ele)
#             for ele2 in integer_list2:
#                 true_label_set.add(ele2)
#             # print("Predictions:", preds)
#             # print("True Labels:", labels)

#     f.write("true_label_set: ")
#     f.write(true_label_set)
#     f.write("predict_label_set: ")
#     f.write(predict_label_set)
#     f.write(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
#     f.write('ZSL: unseen accuracy=%.4f' % (acc1_meter.avg))

#     return acc1_meter.avg

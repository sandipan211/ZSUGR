import torch
import torch.nn.functional as F
import torch.nn as nn


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
            b, _ , _ , _ = inputs.size()
            outputs = model(inputs, is_training=False)
            pred_labels = F.softmax(outputs,-1)
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
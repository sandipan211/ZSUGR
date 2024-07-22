import torch.nn.functional as F
import torch

def loss_gesture_labels(outputs, targets):
    assert 'pred_gesture_logits' in outputs
    src_logits = outputs['pred_gesture_logits']
    # print(f'Output shape of feature for computing loss: {src_logits.shape}')  # [bs, num_seen_classes]
    # print(f'Output shape of target labels for computing loss: {targets.shape}') # [num_seen_classes]
    loss_ges_ce = F.cross_entropy(src_logits, targets)
    return loss_ges_ce

def mimic_loss(produce):

    raw_feature = produce['clip_cls_feature']
    net_feature = produce['gcat_feature']

    return  F.l1_loss(raw_feature, net_feature)
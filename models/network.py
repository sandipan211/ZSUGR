import torch
from .backbone import build_backbone
from ModifiedCLIP import clip
import torch.nn as nn
import numpy as np
from .transformer_encoder import TransformerEncoderLayer, LayerNorm, TransformerEncoder
from .gcat import CrossAttention, CrossAttentionLayer
import pdb

class ZSGR(nn.Module):
    def __init__(self, backbone, args, all_semantics, return_intermediate_dec=False, clip_dim=768, activation="relu", dropout=0.1, normalize_before=False):
        super().__init__()

        self.args = args
        self.split_labels = args.split_labels
        self.backbone = backbone

        # encoder related
        self.enc_layers = args.enc_layers
        self.input_proj = nn.Conv2d(backbone.num_channels, args.cnn_embed_dim, kernel_size=1)
        encoder_layer = TransformerEncoderLayer(args.cnn_embed_dim, args.nhead, args.dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = LayerNorm(args.cnn_embed_dim) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, self.enc_layers, encoder_norm)
        self.encoder_proj = nn.Linear(args.cnn_embed_dim, clip_dim) 
        self.encoder_text_proj = nn.Linear(args.cnn_embed_dim, args.clip_embed_dim) # 256 to 512

        
        # cross-attention related
        cross_attention_norm = LayerNorm(clip_dim)
        cross_attention_layer=CrossAttentionLayer(args.nhead, args.gated_activation, dropout, clip_dim)
        self.gated_cross_attention=CrossAttention(cross_attention_layer, args.num_cross_attention_layers, cross_attention_norm, return_intermediate_dec)        

        # clip related
        self.all_semantics = all_semantics
        self.seen_labels = args.split_labels['train']
        self.unseen_labels = args.split_labels['test_unseen']
        self.seen_semantics = self.all_semantics[self.seen_labels]
        self.unseen_semantics = self.all_semantics[self.unseen_labels]

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.clip_model, self.preprocess = clip.load(self.args.clip_version)
        with torch.no_grad():
            self.clip_visual_proj = self.clip_model.visual.proj.detach()

        self.gesture_class_fc = nn.Sequential(
                nn.Linear(args.clip_embed_dim, args.clip_embed_dim),
                nn.LayerNorm(args.clip_embed_dim),
            )

        if args.with_clip_label:
            # visual_projection is the classifier that maps to num_seen_classes
            if args.fix_clip_label:
                self.visual_projection = nn.Linear(args.clip_embed_dim, len(self.seen_labels), bias=False)
                self.visual_projection.weight.data = self.seen_semantics / self.seen_semantics.norm(dim=-1, keepdim=True)
                # classifier weights are frozen to CLIP knowledge - it will not be trained
                for i in self.visual_projection.parameters():
                    i.require_grads = False
            else:
                self.visual_projection = nn.Linear(args.clip_embed_dim, len(self.seen_labels))
                self.visual_projection.weight.data = self.seen_semantics / self.seen_semantics.norm(dim=-1, keepdim=True)

            # this line might need if-else modification if we integrate supervised learning later
            # eval_projection is the classifier used to classify into seen+unseen classes. 
            # used while testing, and also if we want to train this classifier 

            self.eval_visual_projection = nn.Linear(args.clip_embed_dim, len(self.seen_labels)+len(self.unseen_labels), bias=False)
            self.eval_visual_projection.weight.data = self.all_semantics / self.all_semantics.norm(dim=-1, keepdim=True)
        else:
            self.gesture_class_embedding = nn.Linear(args.clip_embed_dim, len(self.seen_labels))

        # this is a separate classifier initialized with weights of all class semantics to see what results we get using [CLS] token of CLIP
        self.clip_token_cls = self.all_semantics / self.all_semantics.norm(dim=-1, keepdim=True)

        # clip_label=all_semantics, hoi_text=seen_labels, train_clip_label=seen_semantics = \
        #     self.init_classifier_with_CLIP()
        # # num_obj_classes = len(obj_text) - 1  # del nothing
        # self.clip_visual_proj = v_linear_proj_weight

    def forward(self, samples, is_training=True, clip_input=None):

        # step 1: get CNN-based visual features
        # print(samples.device)               -> on CUDA
        features, pos = self.backbone(samples)

        src = self.input_proj(features[-1])
        pos_embed = pos[-1]
        bs, c, h, w = src.shape
        assert (h, w) == (7, 7), "Architecture works with training samples of (224, 224) and with ResNet50 as CNN backbone only"
        # print(f'CNN visual feature shape: {src.shape}')
        # print(f'Positional embedding shape: {pos_embed.shape}')
        # pdb.set_trace()
        src = src.flatten(2).permute(2, 0, 1)                   # src: hw x bs x c (in our case bs=16(in testing),16(in training),c=256,hw should be 49)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)                   # src: hw x bs x c (in our case bs=16(in testing),8(in training),c=256,hw should be 49)
        # print(f'CNN visual feature shape input to vanilla encoder: {src.shape}')
        # print(f'Positional embedding shape input to vanilla encoder: {pos_embed.shape}')
        # pdb.set_trace()

        # step 2: pass CNN feature maps through a vanilla transformer encoder
        memory = self.encoder(src, pos=pos_embed).permute(1, 0, 2) # shape = [B, HW, 256]
        # print(f'Output of vanilla encoder: {memory.shape}')
        # pdb.set_trace()

        # step 3: prepare vanilla encoder input (E) for cross-attention transformer
        if self.args.encoder_only:
            memory_in_clip_dim = self.encoder_text_proj(memory) # output [B,HW,512]
            dtype = memory_in_clip_dim.dtype
            gcat_output = memory_in_clip_dim
        else:
            memory_in_clip_dim = self.encoder_proj(memory) # input 1: vanilla encoder's output = [B, HW, 768]
            dtype = memory_in_clip_dim.dtype

            # step 4: prepare clip contextual output (P) for cross-attention transformer
            clip_cls_feature, clip_context = self.clip_model.encode_image(clip_input)
            clip_cls_feature = clip_cls_feature / clip_cls_feature.norm(dim=1, keepdim=True)
            clip_cls_feature = clip_cls_feature.to(dtype)
            clip_context = clip_context.to(dtype)

            # step 5: feed inputs to gated cross-attention transformer
            # print(f'Output of vanilla encoder being fed to GCAT has shape: {memory_in_clip_dim.shape}')
            # print(f'CLIP contextual feature being fed to GCAT has shape: {clip_context.shape}')
            # pdb.set_trace()
            gcat_output = self.gated_cross_attention(memory_in_clip_dim, clip_context)
            # print(f'GCAT output from transformer: {gcat_output.shape}') # [1, 16, 49, 768]
            gcat_output = gcat_output @ self.clip_visual_proj.to(dtype)
            # print(f'GCAT output after clip visual proj: {gcat_output.shape}') # [1, 16, 49, 512]
        if self.args.with_clip_label:
            logit_scale = self.logit_scale.exp()
            # outputs_final = gcat_output.clone()
            gcat_output = gcat_output / gcat_output.norm(dim=-1, keepdim=True)    

            # print(f"outputs_final : {outputs_final.shape}")

            # the part below is for evaluation time
            if not is_training:
                outputs_gesture_class = logit_scale * self.eval_visual_projection(gcat_output)
                # print(f'GCAT output projected to classes during testing: {outputs_gesture_class.shape}') 

            else:
                # the part below is for training time
                outputs_gesture_class = logit_scale * self.visual_projection(gcat_output)
                # print(f'GCAT output projected to classes during training: {outputs_gesture_class.shape}') # # [1, 16, 49, 10]

        else:
            gcat_output = self.gesture_class_fc(gcat_output)
            # outputs_final = gcat_output.clone()
            outputs_gesture_class = self.gesture_class_embedding(gcat_output)
            # print(f'GCAT output projected to classes using trainable classifier: {outputs_gesture_class.shape}')

        if self.args.encoder_only:
            produce = {
            'pred_gesture_logits': torch.mean(outputs_gesture_class, dim=1),
            'gcat_feature': torch.mean(gcat_output, dim=1)
            }
        else:
            produce = {
                'pred_gesture_logits': torch.mean(outputs_gesture_class[-1], dim=1),
                'clip_cls_feature': clip_cls_feature,
                'clip_contextual_feature': clip_context @ self.clip_visual_proj.to(dtype),
                'gcat_feature': torch.mean(gcat_output[-1], dim=1)
            }

        return produce 

def build(args, all_semantics):

    backbone = build_backbone(args)

    model = ZSGR(
        backbone=backbone,
        args=args,
        all_semantics=all_semantics,
        return_intermediate_dec=True
    )

    return model


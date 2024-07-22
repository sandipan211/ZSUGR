import torch
import torch.nn as nn
from .transformer_encoder import LayerNorm, _get_clones, _get_activation_fn


class CrossAttention(nn.Module):
    def __init__(self, cross_attention_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(cross_attention_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.norm = norm

    
    def forward(self, memory, clip_context):
        output = memory
        intermediate = []


        for _, layer in enumerate(self.layers):
            output,weight,weight_2 = layer(output, clip_context)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
            
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate), weight, weight_2
        
        return output, weight, weight_2

   

class CrossAttentionLayer(nn.Module):
    def __init__(self, nhead, gated_activation, dropout=0.1, gcat_dim=768, E_ftr_dim=49, P_ftr_dim=50):
        super().__init__()

        # TODO: E_ftr_dim and P_ftr_dim depend on the CNN backbone and CLIP backbone versions. Currently hardcoded only for CNN = ResNet50 and CLIP = ViT-B/32. Need to make it more generic.

        self.cross_att_E = nn.MultiheadAttention(gcat_dim, nhead, dropout=dropout, batch_first=True)
        self.cross_att_P = nn.MultiheadAttention(gcat_dim, nhead, dropout=dropout, batch_first=True)
        self.norm_att = LayerNorm(gcat_dim)        
        self.norm_lin = LayerNorm(gcat_dim)        
        self.dropout_att = nn.Dropout(dropout)
        self.dropout_lin1 = nn.Dropout(dropout)
        self.dropout_lin2 = nn.Dropout(dropout)
        self.conv1 = nn.Conv1d(E_ftr_dim, (E_ftr_dim//2), 1, stride=1)
        self.conv2 = nn.Conv1d(P_ftr_dim, (P_ftr_dim//2), 1, stride=1)
        self.activation_att = _get_activation_fn(gated_activation)
        self.activation_lin = _get_activation_fn('relu') 
        self.linear1 = nn.Linear(gcat_dim, 512)
        self.linear2 = nn.Linear(512, gcat_dim)

    def forward(self, memory, clip_context):

        # clip_context -> (bs, 50, 768) and memory -> (hwxbsxc) = (bs, 49, 768)

        # memory path (branch E)
        att_E_output, weight = self.cross_att_E(query=memory, key=clip_context, value=clip_context)
        att_E_output = memory + self.dropout_att(att_E_output) 
        att_E_output = self.norm_att(att_E_output)
        att_E_output = self.activation_att(self.conv1(att_E_output))
        # print(f'Output from branch E: {att_E_output.shape}')  # [bs, 24, 768]
        
        # clip guidance path (branch P)
        att_P_output, weight_2 = self.cross_att_P(query=clip_context, key=memory, value=memory)
        att_P_output = clip_context + self.dropout_att(att_P_output) 
        att_P_output = self.norm_att(att_P_output)
        att_P_output = self.activation_att(self.conv2(att_P_output))
        # print(f'Output from branch P: {att_P_output.shape}') # [bs, 25, 768]

        # gated operations
        gcat_output = memory * torch.cat((att_E_output, att_P_output), 1)

        # feedforward path
        memory = self.linear2(self.dropout_lin1(self.activation_lin(self.linear1(gcat_output))))
        memory = gcat_output + self.dropout_lin2(memory)
        memory = self.norm_lin(memory)
        # print(f'Output of GCAT: {memory.shape}') # [bs, 49, 768]

        return memory, weight, weight_2
    

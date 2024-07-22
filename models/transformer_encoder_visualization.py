import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        try:
            ret = super().forward(x.type(torch.float32))
        except Exception as e:
            print(e)
        return ret.type(orig_type)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, pos):
        output = src

        for layer in self.layers:
            output, weight_encoder = layer(output, pos=pos)

        if self.norm is not None:
            output = self.norm(output)
        # print(f"output shape: {output.shape}")
        return output, weight_encoder

class TransformerEncoderLayer(nn.Module):

    def __init__(self, cnn_embed_dim, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(cnn_embed_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(cnn_embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, cnn_embed_dim)

        self.norm1 = LayerNorm(cnn_embed_dim)
        self.norm2 = LayerNorm(cnn_embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        


    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, pos):
        # print(f"src before q,k : {src.shape}")
        q = k = self.with_pos_embed(src, pos)

        src2, weight_encoder = self.self_attn(q, k, value=src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        # print(f"src after encoder layer q,k : {src.shape}")
        return src, weight_encoder

    def forward_pre(self, src, pos):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2, weight_encoder = self.self_attn(q, k, value=src2)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src, weight_encoder

    def forward(self, src, pos):

        if self.normalize_before:
            return self.forward_pre(src, pos)
        else:
            return self.forward_post(src, pos)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class SwiGlu(nn.Module):
    def __init__(self, parameter):
        super().__init__()    
        self.beta = parameter

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        # beta parameter not used
        return F.silu(gate) * x

def _get_activation_fn(activation, parameter=None):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "sigmoid":
        return F.sigmoid
    if activation == "elu":
        return F.elu
    if activation == "silu":
        return F.silu
    if activation == "SwiGlu":
        return SwiGlu(parameter)
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")        

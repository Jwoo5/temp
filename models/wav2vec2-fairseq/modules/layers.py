import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple

from utils.attention import ScaledDotProductAttention, MultiHeadAttention
from utils.transpose_last import TransposeLast
from utils.fp32_group_norm import Fp32GroupNorm
from utils.fp32_layer_norm import Fp32LayerNorm

class PositionwiseFeedForwardLayer(nn.Module):
    def __init__(self, embed_dim, ffn_dim, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim

        self.fc1 = nn.Linear(self.embed_dim, self.ffn_dim)
        self.fc2 = nn.Linear(self.ffn_dim, self.embed_dim)
        self.active = F.gelu
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        output = self.fc1(inputs)
        output = self.active(output)
        output = self.dropout(output)
        output = self.fc2(output)
        return output

class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, ffn_dim, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_head = n_heads
        self.d_head = embed_dim // n_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout

        self.self_attn = MultiHeadAttention(self.embed_dim, self.n_head, dropout)
        self.norm_layer1 = nn.LayerNorm(self.embed_dim, eps = 1e-6)
        self.pos_ffn = PositionwiseFeedForwardLayer(self.embed_dim, self.ffn_dim, dropout)
        self.norm_layer2 = nn.LayerNorm(self.embed_dim, eps = 1e-6)
    
    def forward(self, inputs):
        att_outputs = self.self_attn(inputs, inputs, inputs)
        att_outputs = self.norm_layer1(att_outputs + inputs)

        ffn_outputs = self.pos_ffn(att_outputs)
        ffn_outputs = self.norm_layer2(ffn_outputs + att_outputs)

        return ffn_outputs

class Encoder(nn.Module):
    def __init__(self, n_layer, embed_dim, n_heads, ffn_dim, dropout):
        super().__init__()
        self.n_layer = n_layer
        self.embed_dim = embed_dim
        self.n_head = n_heads
        self.d_head = embed_dim // n_heads
        self.ffn_dim = ffn_dim
        
        self.layers = nn.ModuleList([EncoderLayer(self.embed_dim, self.n_head, self.ffn_dim, dropout) for _ in range(self.n_layer)])

    def forward(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

class ConvFeatureExtraction(nn.Module):
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        in_d: int = 1,
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

        def block(
            n_in,
            n_out,
            k,
            stride,
            is_layer_norm = False,
            is_group_norm = False,
            conv_bias = False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride = stride, bias = conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv
            
            assert (
                is_layer_norm and is_group_norm
            ) == False, "layer norm and group norm ar exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, dim, affine = True),
                        nn.GELU(),
                    )
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm = mode == "layer_norm",
                    is_group_norm = mode == "default" and i == 0,
                    conv_bias = conv_bias,
                )
            )
            in_d = dim
    
    def forward(self, x):
        # B x T -> B x C x T
        if len(x.shape) < 3:
            x = x.unsqueeze(1)

        for conv in self.conv_layers:
            x = conv(x)

        return x
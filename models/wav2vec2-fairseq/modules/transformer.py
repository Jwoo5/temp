import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.same_pad import SamePad
from utils.init_bert_params import init_bert_params
from utils.fp32_layer_norm import LayerNorm
from .layers import Encoder, ConvFeatureExtraction

class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.dropout = args.dropout
        self.embedding_dim = args.embed_dim
        
        self.pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size = args.conv_pos,
            padding = args.conv_pos // 2,
            groups = args.conv_pos_groups,
        )
        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (args.conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean = 0, std = std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name = "weight", dim = 2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(args.conv_pos), nn.GELU())

        self.encoder = Encoder(
            n_layer = args.n_layers,
            embed_dim = self.embedding_dim,
            n_heads = args.n_heads,
            ffn_dim = args.ffn_dim,
            dropout = args.dropout
        )

        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)

        self.layerdrop = args.layer_dropout

        self.apply(init_bert_params)
    
    def extract_features(self, x, padding_mask = None):
        if padding_mask is not None:
            x[padding_mask] = 0
        
        x_conv = self.pos_conv(x.transpose(1,2))
        x_conv = x_conv.transpose(1,2)
        x += x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training = self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        x = self.encoder(x)
        
        # T x B x C -> B x T x C
        x = x.transpose(0,1)

        return x
    
    def forward(self, x, padding_mask = None):
        x = self.extract_features(x, padding_mask)

        if self.layer_norm_first:
            x = self.layer_norm(x)
        
        return x
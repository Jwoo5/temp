import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.attention import ScaledDotProductAttention, MultiHeadAttention

class PositionwiseFeedForwardLayer(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.fc1 = nn.Linear(self.d_model, self.d_ff)
        self.fc2 = nn.Linear(self.d_ff, self.d_model)
        self.active = F.gelu
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        output = self.fc1(inputs)
        output = self.active(output)
        output = self.dropout(output)
        output = self.fc2(output)
        return output

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_head, d_ff, dropout):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.d_ff = d_ff
        self.dropout = dropout

        self.self_attn = MultiHeadAttention(self.d_model, self.n_head, self.d_head, dropout)
        self.norm_layer1 = nn.LayerNorm(self.d_model, eps = 1e-6)
        self.pos_ffn = PositionwiseFeedForwardLayer(self.d_model, self.d_ff, dropout)
        self.norm_layer2 = nn.LayerNorm(self.d_model, eps = 1e-6)
    
    def forward(self, inputs):
        att_outputs = self.self_attn(inputs, inputs, inputs)
        att_outputs = self.norm_layer1(att_outputs + inputs)

        ffn_outputs = self.pos_ffn(att_outputs)
        ffn_outputs = self.norm_layer2(ffn_outputs + att_outputs)

        return ffn_outputs

class Encoder(nn.Module):
    def __init__(self, n_layer, d_model, n_head, d_head, d_ff, dropout):
        super().__init__()
        self.n_layer = n_layer
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.d_ff = d_ff
        
        self.layers = nn.ModuleList([EncoderLayer(self.d_model, self.n_head, self.d_head, self.d_ff, dropout) for _ in range(self.n_layer)])

    def forward(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

class Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        
        # n_seq : 187 for mitbih_refined

        # (batch_size, 1, 187) -> (batch_size, 8, 93)
        self.conv1 = nn.Conv1d(in_channels = 1, out_channels = 64, kernel_size = 3, stride = 2, padding = 0)
        # (batch_size, 8, 93) -> (batch_size, 16, 93)
        self.conv2 = nn.Conv1d(in_channels = 64, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)
        # (batch_size, 16, 93) -> (batch_size, 32, 93)
        self.conv3 = nn.Conv1d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)
        self.active = F.relu
    
    def forward(self, inputs):
        # inputs : (batch_size, n_seq)
        inputs = inputs[:,:,np.newaxis].transpose(1,2)
        outputs = self.active(self.conv1(inputs))
        outputs = self.active(self.conv2(outputs))
        outputs = self.active(self.conv3(outputs))
        return outputs
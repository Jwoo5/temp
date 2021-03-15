import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_head, dropout):
        super().__init__()
        self.d_head = d_head
        self.dropout = nn.Dropout(dropout)
        self.scale = 1 / (self.d_head ** 0.5)
    
    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-1,-2)).mul_(self.scale)
        attn_prob = F.softmax(scores, dim = -1)
        attn_prob = self.dropout(attn_prob)
        context = torch.matmul(attn_prob, V)

        return context

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, d_head, dropout):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.W_Q = nn.Linear(self.d_model, self.n_head * self.d_head)
        self.W_K = nn.Linear(self.d_model, self.n_head * self.d_head)
        self.W_V = nn.Linear(self.d_model, self.n_head * self.d_head)
        self.scaled_dot_attn = ScaledDotProductAttention(self.d_head, dropout)
        self.linear = nn.Linear(self.n_head * self.d_head, self.d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V):
        # (batch_size, n_seq, d_model)
        batch_size = Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_head, self.d_head).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_head, self.d_head).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_head, self.d_head).transpose(1,2)

        context = self.scaled_dot_attn(q_s, k_s, v_s)
        context = context.transpose(1,2).contiguous().view(batch_size, -1, self.n_head * self.d_head)

        output = self.linear(context)
        output = self.dropout(output)

        # print(output.shape)

        return output
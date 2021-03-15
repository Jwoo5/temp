from .layers import Encoder, Embedding

import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, n_layer, d_model, n_head, d_head, d_ff, n_classes, dropout):
        super().__init__()
        self.n_layer = n_layer
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.d_ff = d_ff
        self.n_classes = n_classes

        self.embedding = Embedding()
        self.encoder = Encoder(self.n_layer, self.d_model, self.n_head, self.d_head, self.d_ff, dropout)
        self.fc1 = nn.Linear(47616, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, self.n_classes)
        self.active = F.relu

    def forward(self, inputs):
        # input : (batch_size, n_seq)

        # (batch_size, d_model, n_seq) -> (batch_size, n_seq, d_model)
        outputs = self.embedding(inputs).transpose(1,2)
        # debug
        # print(outputs.shape)
        # (batch_size, n_seq, d_model)
        outputs = self.encoder(outputs)
        outputs = outputs.view(outputs.size(0), -1)

        outputs = self.active(self.fc1(outputs))
        outputs = self.active(self.fc2(outputs))
        outputs = self.fc3(outputs)
        outputs = F.softmax(outputs, dim = -1)

        return outputs
    
    def save(self, epoch, loss, score, path):
        torch.save({
            "epoch" : epoch,
            "loss" : loss,
            "score": score,
            "state_dict" : self.state_dict()
        }, path)
    
    def load(self, path):
        save = torch.load(path)
        self.load_state_dict(save["state_dict"])
        return save["epoch"], save["loss"], save["score"]
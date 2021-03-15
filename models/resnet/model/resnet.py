import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        self.conv1 = nn.Conv1d(in_channels = self.d_model, out_channels = self.d_model, kernel_size = 5, padding = 2)
        self.conv2 = nn.Conv1d(in_channels = self.d_model, out_channels = self.d_model, kernel_size = 5, padding = 2)
        self.pool = nn.MaxPool1d(kernel_size = 5, stride = 2)
        self.activation = F.relu

    def forward(self, input):
        output = self.conv1(input)
        output = self.activation(output)
        output = self.conv2(output)

        output = input + output
        output = self.activation(output)
        output = self.pool(output)

        return output

class Resnet(nn.Module):
    def __init__(self, d_in, d_model, n_layer, n_classes):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_classes = n_classes

        self.conv1 = nn.Conv1d(in_channels = self.d_in, out_channels = self.d_model, kernel_size = 5, padding = 2)
        self.layers = nn.ModuleList([Block(self.d_model) for _ in range(self.n_layer)])
        self.fc1 = nn.Linear(d_model * 2, d_model)
        self.fc2 = nn.Linear(d_model, self.n_classes)
        self.activate = F.relu
    
    def forward(self, input):
        batch_size = input.shape[0]
        
        input = input[:,:,np.newaxis].transpose(1,2)
        output = self.conv1(input)
        for layer in self.layers:
            output = layer(output)

        output = torch.flatten(output, start_dim = 1)
        output = self.activate(self.fc1(output))
        output = self.fc2(output)
        output = F.softmax(output, dim = -1)

        return output

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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functinal as F

# TODO conv_block parameters, ... -> args
# ex) args.embed_dim, args.conv, ...

class CLOCS(nn.module):
    def __init__(self, args):
        super().__init__()

        def conv_block(kernel_size, in_channel, out_channel):
            return nn.Sequential(nn.Conv1d(in_channels = in_channel, out_channels = out_channel, kernel_size = kernel_size, stride = 3),
                                nn.BatchNorm1d(out_channel),
                                nn.ReLU(),
                                nn.MaxPool1d(kernel_size = 2),
                                nn.Dropout(0.1))
        
        self.embed_dim = args.embed_dim
        self.n_classes = args.n_classes
        self.conv_layers = eval(args.conv_layers)

        self.conv_blocks = [conv_block(k,i,o) for k,i,o in self.conv_layers]

        # XXX should be modified
        self.fc1 = nn.Linear(self.conv_layers[-1,-1], self.embed_dim)
        self.fc2 = nn.Linear(self.embed_dim, self.n_classes)

    def forward(self, x):
        output = x
        
        for block in self.conv_blocks:
            output = block(output)
        
        output = self.fc1(output)
        output = nn.ReLU(output)
        output = self.fc2(output)

        return output
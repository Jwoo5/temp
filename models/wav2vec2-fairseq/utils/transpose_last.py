import torch.nn as nn

class TransposeLast(nn.Module):
    def __init__(self, deconstruct_idx = None):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx
    
    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(-2,-1)
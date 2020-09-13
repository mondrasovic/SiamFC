import torch
import torch.nn as nn

class LogisticLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y):
        return torch.log(1 + torch.exp(-y * x))

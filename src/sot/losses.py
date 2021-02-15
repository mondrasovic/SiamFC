import torch
import torch.nn as nn


class WeightedBCELoss(nn.Module):
    def __init__(self, weight_mat: torch.Tensor) -> None:
        super().__init__()
        
        self.loss = nn.BCEWithLogitsLoss(weight=weight_mat, reduction='sum')
    
    def forward(
            self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(pred, target)

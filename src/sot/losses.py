#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Milan Ondrasovic <milan.ondrasovic@gmail.com>

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCELoss(nn.Module):
    def __init__(self, weight_mat: torch.Tensor) -> None:
        super().__init__()
        
        self.weight_mat: torch.Tensor = weight_mat
    
    def forward(
            self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            pred, target, weight=self.weight_mat, reduction='sum')

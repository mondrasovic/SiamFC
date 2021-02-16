#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Milan Ondrasovic <milan.ondrasovic@gmail.com>

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class _ConvBlock(nn.Module):
    def __init__(
            self, in_channels: int, out_channels: int, *, kernel_size: int = 3,
            stride: int = 1, groups: int = 1, activation: bool = True,
            max_pool: bool = False) -> None:
        super().__init__()
        
        modules = [
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=kernel_size, stride=stride, groups=groups,
                bias=True)]
        
        if activation:
            modules.append(nn.BatchNorm2d(
                num_features=out_channels, eps=1e-6, momentum=0.05))
            modules.append(nn.ReLU(inplace=True))
        
        if max_pool:
            modules.append(nn.MaxPool2d(kernel_size=3, stride=2))
        
        self.block = nn.Sequential(*modules)
    
    def forward(self, x):
        return self.block(x)


class SiamFCModel(nn.Module):
    def __init__(self, response_map_scale: float = 0.001) -> None:
        super().__init__()
        
        self.response_map_scale: float = response_map_scale
        
        self.conv1 = _ConvBlock(
            in_channels=3, out_channels=96, kernel_size=11, stride=2,
            max_pool=True)
        self.conv2 = _ConvBlock(
            in_channels=96, out_channels=256, kernel_size=5, groups=2,
            max_pool=True)
        self.conv3 = _ConvBlock(in_channels=256, out_channels=384)
        self.conv4 = _ConvBlock(in_channels=384, out_channels=384)
        self.conv5 = _ConvBlock(
            in_channels=384, out_channels=256, groups=2, activation=False)
        
        self._initialize_weights()
    
    def forward(
            self, exemplar: torch.Tensor,
            instance: torch.Tensor) -> torch.Tensor:
        exemplar_emb = self.extract_visual_features(exemplar)
        instance_emb = self.extract_visual_features(instance)
        response_map = self.calc_response_map(exemplar_emb, instance_emb)
        return response_map
    
    def extract_visual_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x
    
    def calc_response_map(
            self, exemplar_emb: torch.Tensor,
            instance_emb: torch.Tensor) -> torch.Tensor:
        response_map = self.cross_corr(exemplar_emb, instance_emb)
        return response_map * self.response_map_scale
    
    @staticmethod
    def cross_corr(
            exemplar_emb: torch.Tensor,
            instance_emb: torch.Tensor) -> torch.Tensor:
        n_exemplars = exemplar_emb.shape[0]
        n_instances, instance_c, instance_h, instance_w = instance_emb.shape
        
        instance_emb = instance_emb.reshape(
            -1, n_exemplars * instance_c, instance_h, instance_w)
        response_map = F.conv2d(
            input=instance_emb, weight=exemplar_emb, groups=n_exemplars)
        response_map = response_map.reshape(
            n_instances, -1, response_map.shape[-2], response_map.shape[-1])
        
        return response_map
    
    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_normal_(
                    module.weight.data, mode='fan_out', nonlinearity='relu')
                module.bias.data.fill_(0)
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

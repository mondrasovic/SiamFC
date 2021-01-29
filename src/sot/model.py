import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    def __init__(
            self, in_channels: int, out_channels: int, *, kernel_size: int = 3,
            stride: int = 1, groups: int = 1, max_pool: bool = False) -> None:
        super().__init__()
        
        modules = [
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=kernel_size, stride=stride, groups=groups,
                bias=True),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)]
        
        if max_pool:
            modules.append(nn.MaxPool2d(kernel_size=3, stride=2))

        self.block = nn.Sequential(*modules)

    def forward(self, x):
        return self.block(x)


class SiamFCModel(nn.Module):
    def __init__(self, response_map_scale: float = 0.001) -> None:
        super().__init__()
        
        assert response_map_scale > 0, "response map scale must be positive"
        
        self.response_map_scale: float = response_map_scale
        
        self.conv1 = _ConvBlock(
            in_channels=3, out_channels=96, kernel_size=11, stride=2,
            max_pool=True)
        self.conv2 = _ConvBlock(
            in_channels=96, out_channels=256, kernel_size=5, groups=2,
            max_pool=True)
        self.conv3 = _ConvBlock(in_channels=256, out_channels=384)
        self.conv4 = _ConvBlock(in_channels=384, out_channels=384)
        self.conv5 = _ConvBlock(in_channels=384, out_channels=32, groups=2)
        
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
        assert exemplar_emb.ndim == instance_emb.ndim
        assert exemplar_emb.ndim == 4

        n_instances, instance_c, instance_h, instance_w = instance_emb.shape

        instance_emb = instance_emb.view(1, -1, instance_h, instance_w)
        response_map = F.conv2d(
            input=instance_emb, weight=exemplar_emb, groups=n_instances)
        response_map = response_map.view(
            n_instances, -1, response_map.shape[-2], response_map.shape[-1])
        
        return response_map

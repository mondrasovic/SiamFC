import torch.nn as nn
import torch.nn.functional as F

class _ConvBlock(nn.Module):
    def __init__(
            self, in_channels, out_channels, *, kernel_size=3, stride=1,
            groups=1, max_pool=False):
        super().__init__()
        
        modules = [
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=kernel_size, stride=stride, groups=groups),
            nn.BatchNorm2d(num_features=out_channels, eps=1.e-6, momentum=0.05),
            nn.ReLU(inplace=True)]
        
        if max_pool:
            modules.append(nn.MaxPool2d(kernel_size=3, stride=2))

        self.block = nn.Sequential(*modules)

    def forward(self, x):
        return self.block(x)

class SiamFC(nn.Module):
    OUTPUT_STRIDE = 8
    
    def __init__(self):
        super().__init__()
        
        self.conv1 = _ConvBlock(
            in_channels=3, out_channels=96, kernel_size=11, stride=2,
            max_pool=True)
        self.conv2 = _ConvBlock(
            in_channels=96, out_channels=256, kernel_size=5, groups=2,
            max_pool=True)
        self.conv3 = _ConvBlock(in_channels=256, out_channels=384)
        self.conv4 = _ConvBlock(in_channels=384, out_channels=384)
        self.conv5 = _ConvBlock(in_channels=384, out_channels=256, groups=2)
        
    def forward(self, z, x):
        z_embed = self.extract_visual_features(z)
        x_embed = self.extract_visual_features(x)
        response_map = self._cross_corr(z_embed, x_embed)
        return response_map
    
    def extract_visual_features(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x
    
    @staticmethod
    def _cross_corr(z_embed, x_embed, *, scale=0.001):
        n_exemplars = z_embed.size(0)
        n_instances, instance_c, instance_h, instance_w = x_embed.size()
        
        x_embed = x_embed.view(
            -1, n_exemplars * instance_c, instance_h, instance_w)
        y = F.conv2d(x_embed, z_embed, groups=n_exemplars)
        y = y.view(n_exemplars, -1, y.size(-2), y.size(-1))
        y = y * scale
        
        return y

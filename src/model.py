import torch.nn as nn
import torch.nn.functional as F

class _ConvBlock(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, max_pool=False):
        super().__init__()
        
        modules = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                groups=groups),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        ]
        
        if max_pool:
            modules.append(nn.MaxPool2d(kernel_size=3, stride=2))

        self._block = nn.Sequential(*modules)

    def forward(self, x):
        return self._block(x)

class SiamFC(nn.Module):
    OUTPUT_STRIDE = 8
    
    def __init__(self):
        super().__init__()
        
        self._conv1 = _ConvBlock(
            in_channels=3,
            out_channels=96,
            kernel_size=11,
            stride=2,
            max_pool=True)
        self._conv2 = _ConvBlock(
            in_channels=96,
            out_channels=256,
            kernel_size=5,
            groups=2,
            max_pool=True)
        
        self._conv3 = _ConvBlock(in_channels=256, out_channels=384)
        self._conv4 = _ConvBlock(in_channels=384, out_channels=384)
        self._conv5 = _ConvBlock(in_channels=384, out_channels=256, groups=2)
        
    def forward(self, x_exemplar, x_search):
        exemplar_emb = self.extract_visual_features(x_exemplar)
        search_emb = self.extract_visual_features(x_search)

        cross_corr = self._cross_corr(search_emb, exemplar_emb)
        
        return cross_corr
    
    def extract_visual_features(self, x):
        x = self._conv1(x)
        x = self._conv2(x)
        x = self._conv3(x)
        x = self._conv4(x)
        x = self._conv5(x)
        
        return x
    
    @staticmethod
    def _cross_corr(search_emb, exemplar_emb):
        n_exemplars = exemplar_emb.size(0)
        n_searches, search_c, search_h, search_w = search_emb.size()
        
        search_emb = search_emb.view(-1, n_exemplars * search_c, search_h, search_w)
        y = F.conv2d(search_emb, exemplar_emb, groups=n_exemplars)
        y = y.view(n_exemplars, -1, y.size(-2), y.size(-1))
        
        return y

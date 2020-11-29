import torch
import numpy as np
from torch import optim

from PIL import Image
from got10k import trackers
from typing import Optional

from .common import TrackerConfig, BBoxT
from .model import SiamFC


class TrackerSiamFC(trackers.Tracker):
    def __init__(
            self, config: TrackerConfig,
            model_path: Optional[str] = None) -> None:
        super().__init__(name='SiamFC', is_deterministic=True)
        
        self.config = config
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = SiamFC()
        if model_path is not None:
            self.model.load_state_dict(
                torch.load(
                    model_path, map_location=lambda storage, location: storage))
        self.model = self.model.to(self.device)
        
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=self.config.initial_lr,
            weight_decay=self.config.weight_decay,
            momentum=self.config.momentum)
        self.criterion = None
        
        # Learning rate is geometrically annealed at each epoch. Starting from
        # A and terminating at B, then the known gamma factor x for n epochs
        # is computed as
        #         A * x ^ n = B,
        #                 x = (B / A)^(1 / n).
        gamma = np.power(
            self.config.ultimate_lr / self.config.initial_lr,
            1.0 / self.config.n_epochs)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma)
        
        self.z_center_yx = None
        self.z_size_hw = None
        self.z_with_context_size = None
        self.x_size_adjusted = None
        
        # Create a normalized cosine (Hanning) window.
        self.upscaled_response_size = (self.config.response_size *
                                       self.config.response_upscale)
        self.hanning = np.outer(
            np.hanning(self.upscaled_response_size),
            np.hanning(self.upscaled_response_size))
        self.hanning /= np.sum(self.hanning)
        
        # Authors chose to search for the object over multiple different scales.
        n_half_search_scales = self.config.n_scales // 2
        self.scales = self.config.scale_step ** np.linspace(
            -n_half_search_scales, n_half_search_scales,
            self.config.n_scales)
    
    @torch.no_grad()
    def init(self, img: Image, bbox: BBoxT) -> None:
        self.model.eval()
        
        # Convert the bounding box to 0-indexed and center-based with
        # [y, x, h, w] ordering.
        bbox_yxhw = np.float32(
            ((bbox[1] - 1) + (bbox[3] - 1) / 2.0,
             (bbox[0] - 1) + (bbox[2] - 1) / 2.0, bbox[3], bbox[2]))
        self.z_center_yx = bbox_yxhw[:2]
        self.z_size_hw = bbox_yxhw[2:]
        
        # TODO Refactor this.
        # Exemplar and instance (search) sizes.
        # The endeavor is to resize the image so that the bounding box plus the
        # margin have a fixed area. In the original paper, the constraint was
        #         s(w + 2p)s(h + 2p) = A,
        # where p = (w + h) / 4, in other words, half of the average dimension,
        # and A = 127^2. However, it can be implemented as
        #         s = sqrt(A / ((w + p)(h + p)),
        # given p = (w + h). The multiplication by 2 essentially cancels out
        # the "half" in p.
        context_size = (self.config.context *
                        np.sum(self.z_size_hw))  # Average dimension.
        self.z_with_context_size = np.sqrt(
            np.prod(self.z_size_hw + context_size))
        scale = self.z_with_context_size / self.config.exemplar_size
        self.x_size_adjusted = round(self.config.instance_size * scale)
    
    @torch.no_grad()
    def update(self, image: Image) -> None:
        self.model.eval()

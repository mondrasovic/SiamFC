import dataclasses

from typing import Optional

import torch
import numpy as np
from torch import optim

from PIL import Image

from got10k import trackers

from .datasets import BBoxT
from .model import SiamFC


@dataclasses.dataclass(frozen=True)
class TrackerConfig:
    # Training parameters.
    n_epochs: int = 50
    batch_size: int = 8
    
    # Optimizer parameters.
    weight_decay: float = 5e-4
    momentum: float = 0.9
    initial_learning_rate: float = 1e-2
    ultimate_learning_rate: float = 1e-5
    
    # Dataset parameters.
    pairs_per_sequence: int = 10
    exemplar_size: int = 127
    instance_size: int = 255
    max_pair_distance: int = 100  # Distance in the no. of frames between exemplar and instance.
    context: float = 0.5
    min_bbox_area: int = 10
    
    # Tracker inference parameters.
    response_size: int = 17  # Dimension length of the square output score (response) map.
    response_upscale: int = 272 // 17  # Upscale coefficient for the response map resizing. Authors
                                       # chose to upscale the response map from 17x17 to 272x272.
    n_search_scales: int = 5  # No. of different scaled over which the object is searched for.
    scale_step: float = 1.025
    scale_damping: float = 0.35  # Scale interpolation coefficient to provide damping.

class TrackerSiamFC(trackers.Tracker):
    def __init__(self, config: TrackerConfig, model_path: Optional[str] = None) -> None:
        super().__init__(name='SiamFC', is_deterministic=True)
        
        self._config = config
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self._model = SiamFC()
        if model_path is not None:
            self._model.load_state_dict(
                torch.load(model_path, map_location=lambda storage, location: storage)
            )
        self._model = self._model.to(self._device)
        
        self._optimizer = optim.SGD(
            self._model.parameters(),
            lr=self._config.initial_learning_rate,
            weight_decay=self._config.weight_decay,
            momentum=self._config.momentum
        )
        self._criterion = None
        
        # Learning rate is geometrically annealed at each epoch. Starting from A and terminating at
        # B, then the known gamma factor x for n epochs is computed as
        #         A * x ^ n = B,
        #                 x = (B / A)^(1 / n).
        gamma = np.power(
            self._config.ultimate_learning_rate / self._config.initial_learning_rate,
            1.0 / self._config.n_epochs
        )
        self._lr_scheduler = optim.lr_scheduler.ExponentialLR(self._optimizer, gamma)
        
        self._exemplar_center = None
        self._exemplar_size = None
        self._exemplar_with_context_size = None
        self._instance_size_adjusted = None
        
        # Create a normalized cosine (Hanning) window.
        self._upscaled_response_size = self._config.response_size * self._config.response_upscale
        self._hanning_window = np.outer(
            np.hanning(self._upscaled_response_size), np.hanning(self._upscaled_response_size)
        )
        self._hanning_window /= np.sum(self._hanning_window)
        
        # Authors chose to search for the object over multiple different scales.
        n_half_search_scales = self._config.n_search_scales // 2
        self._scale_factors = self._config.scale_step ** np.linspace(
            -n_half_search_scales, n_half_search_scales, self._config.n_search_scales
        )
    
    @torch.no_grad()
    def init(self, image: Image, bbox: BBoxT) -> None:
        self._model.eval()
        
        # Convert the bounding box to 0-indexed and center-based with [y, x, h, w] ordering.
        bbox_yxhw = np.float32(
            ((bbox[1] - 1) + (bbox[3] - 1) / 2.0,
             (bbox[0] - 1) + (bbox[2] - 1) / 2.0,
             bbox[3],
             bbox[2]))
        self._exemplar_center = bbox_yxhw[:2]
        self._exemplar_size = bbox_yxhw[2:]
        
        # TODO Refactor this.
        # Exemplar and instance (search) sizes.
        # The endeavor is to resize the image so that the bounding box plus the margin have a fixed
        # area. In the original paper, the constraint was
        #         s(w + 2p)s(h + 2p) = A,
        # where p = (w + h) / 4, in other words, half of the average dimension, and A = 127^2.
        # However, it can be implemented as
        #         s = sqrt(A / ((w + p)(h + p)),
        # given p = (w + h). The multiplication by 2 essentially cancels out the "half" in p.
        context_size = self._config.context * np.sum(self._exemplar_size)  # Average dimension.
        self._exemplar_with_context_size = np.sqrt(np.prod(self._exemplar_size + context_size))
        scale = self._exemplar_with_context_size / self._config.exemplar_size
        self._instance_size_adjusted = round(self._config.instance_size * scale)
        
        
        
    
    @torch.no_grad()
    def update(self, image: Image) -> None:
        self._model.eval()

import dataclasses

import torch
import numpy as np

from numbers import Number
from typing import Union, Tuple

BBoxT = Union[Tuple[Number, Number, Number, Number], np.ndarray]
TensorT = torch.tensor


@dataclasses.dataclass(frozen=True)
class TrackerConfig:
    # Training parameters.
    n_epochs: int = 50
    batch_size: int = 8
    
    # Optimizer parameters.
    weight_decay: float = 5e-4
    momentum: float = 0.9
    initial_lr: float = 1e-2
    ultimate_lr: float = 1e-5
    
    # Dataset parameters.
    pairs_per_seq: int = 10
    exemplar_size: int = 127
    instance_size: int = 255
    max_pair_dist: int = 100  # No. of frames between exemplar and instance.
    context: float = 0.5
    min_bbox_area: int = 10
    
    # Inference parameters.
    response_size: int = 17  # Dimension of the output response (score) map.
    response_upscale: int = 272 // 17  # Upscale coefficient for the response
    # map. Authors chose to upscale the
    # response map from 17x17 to 272x272.
    n_scales: int = 5  # No. of different scales for object search.
    scale_step: float = 1.025
    scale_damping: float = 0.35  # Scale interpolation (for damping).

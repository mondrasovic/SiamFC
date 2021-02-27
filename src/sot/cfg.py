#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Milan Ondrasovic <milan.ondrasovic@gmail.com>

import dataclasses


@dataclasses.dataclass(frozen=True)
class TrackerConfig:
    # Training parameters.
    n_epochs: int = 50
    n_epochs_val: int = 0
    batch_size: int = 16
    n_workers: int = 6
    free_cpus: int = 0
    
    # Optimizer parameters.
    weight_decay: float = 5e-4
    momentum: float = 0.9
    initial_lr: float = 1e-2
    ultimate_lr: float = 1e-5
    
    # Dataset parameters.
    # Number of pairs per each tracking sequence.
    pairs_per_seq: int = 1
    # Exemplar (initial template) image side size.
    exemplar_size: int = 127
    # Search (future) image side size.
    instance_size: int = 255
    
    # No. of frames between exemplar and instance.
    max_pair_dist: int = 100
    
    # Minimum area of the bounding box.
    min_bbox_area: int = 20
    
    # Inference parameters.
    response_size: int = 17  # Dimension of the output response (score) map.
    
    # Influence of cosine (Hanning) window on the response map.
    cosine_win_influence: float = 0.176
    
    # Total stride of the network. It is used when mapping from the response
    # map position to the image position.
    total_stride: int = 8
    
    # Radius in the basic response map as which denotes which cells should be
    # marked as 1, i.e., a positive class. All other cells are then considered
    # to belong to negative class, hence their value is 0. This radius is
    # measured from the center.
    positive_class_radius: int = 16
    
    # Upscale coefficient for the response map.
    # Change of size from 17x17 to 272x272.
    response_upscale: int = 272 // 17
    
    # Scaling coefficient to multiply the cells of the response map with.
    response_map_scale: float = 0.001
    
    # No. of different scales for object search.
    n_scales: int = 3
    
    # Change in scale to search over.
    scale_step: float = 1.0375  # 1.025
    
    # Scale linear interpolation (for smooth transition).
    scale_damping: float = 0.59
    
    # Penalty term for searching in different scale factors other than 1.
    # For example, if searching in 5 different scales, such as
    # 0.8, 0.9, 1, 1.1, 1.2, then all these response maps except for the one in
    # the middle would be additionally multiplied by this penalty term.
    scale_penalty: float = 0.9745

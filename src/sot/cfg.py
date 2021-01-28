import dataclasses


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

    # No. of frames between exemplar and instance.
    max_pair_dist: int = 100
    min_bbox_area: int = 10
    
    # Inference parameters.
    response_size: int = 17  # Dimension of the output response (score) map.
    
    # Influence of cosine (Hanning) window on the response map.
    cosine_win_influence = 0.176
    
    total_stride = 8
    
    # Upscale coefficient for the response
    # map. Authors chose to upscale the
    # response map from 17x17 to 272x272.
    response_upscale: int = 272 // 17

    # No. of different scales for object search.
    n_scales: int = 5
    scale_step: float = 1.025
    # Scale interpolation (for damping).
    scale_damping: float = 0.35

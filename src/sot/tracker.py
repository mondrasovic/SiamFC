import torch
import cv2 as cv
import numpy as np
from torch import optim

from typing import Optional, Iterable

from sot.cfg import TrackerConfig
from model import SiamFC
from utils import (
    center_crop_and_resize, cv_img_to_tensor, calc_bbox_side_size_with_context)
from bbox import BBox


class TrackerSiamFC:
    def __init__(
            self, config: TrackerConfig,
            model_path: Optional[str] = None) -> None:
        self.cfg: TrackerConfig = config
        self.device: torch.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model: SiamFC = SiamFC()
        if model_path is not None:
            self.model.load_state_dict(
                torch.load(
                    model_path, map_location=lambda storage, location: storage))
        self.model = self.model.to(self.device)
        
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay,
            momentum=self.cfg.momentum)
        self.criterion = None
        
        # Learning rate is geometrically annealed at each epoch. Starting from
        # A and terminating at B, then the known gamma factor x for n epochs
        # is computed as
        #         A * x ^ n = B,
        #                 x = (B / A)^(1 / n).
        gamma = np.power(
            self.cfg.ultimate_lr / self.cfg.initial_lr,
            1.0 / self.cfg.n_epochs)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma)
        
        # Create a normalized cosine (Hanning) window.
        self.upscaled_response_size: int =\
            self.cfg.response_size * self.cfg.response_upscale
        self.hanning: np.ndarray = np.outer(
            np.hanning(self.upscaled_response_size),
            np.hanning(self.upscaled_response_size))
        self.hanning /= np.sum(self.hanning)
        
        # Authors chose to search for the object over multiple different scales.
        n_half_search_scales = self.cfg.n_scales // 2
        self.search_scales: np.ndarray = self.cfg.scale_step ** np.linspace(
            -n_half_search_scales, n_half_search_scales,
            self.cfg.n_scales)
        
        self.instance_bbox: Optional[BBox] = None
        self.exemplar_bbox: Optional[BBox] = None
        self.exemplar_img: Optional[np.ndarray] = None
        self.exemplar_emb: Optional[np.ndarray] = None
    
    @torch.no_grad()
    def init(self, img: np.ndarray, bbox: np.ndarray) -> None:
        assert img.ndim == 3
        assert (len(bbox) == 4) and (bbox.ndim == 1)
        
        self.model.eval()

        # Exemplar and instance (search) sizes.
        # The endeavor is to resize the image so that the bounding box plus the
        # margin have a fixed area. In the original paper, the constraint was
        #         s(w + 2p)s(h + 2p) = A,
        # where p = (w + h) / 4, in other words, half of the average dimension,
        # and A = 127^2. However, it can be implemented as
        #         s = sqrt(A / ((w + p)(h + p)),
        # given p = (w + h). The multiplication by 2 essentially cancels out
        # the "half" in p.
        
        assert self.cfg.exemplar_size < self.cfg.instance_size
        bbox = BBox(*bbox)
        instance_side_size = calc_bbox_side_size_with_context(bbox)
        size_ratio = self.cfg.exemplar_size / self.cfg.instance_size
        exemplar_side_size = instance_side_size * size_ratio
        
        self.exemplar_bbox = BBox.build_from_center_and_size(
            bbox.center, np.asarray((exemplar_side_size, exemplar_side_size)))
        self.exemplar_img = center_crop_and_resize(
            img, self.exemplar_bbox,
            (self.cfg.exemplar_size, self.cfg.exemplar_size))
        
        exemplar_img_tensor = cv_img_to_tensor(self.exemplar_img, self.device)
        self.exemplar_emb: torch.Tensor = self.model.extract_visual_features(
            exemplar_img_tensor)
    
    @torch.no_grad()
    def update(self, img: np.ndarray) -> None:
        assert img.ndim == 3
        
        self.model.eval()
        
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        instance_size = (self.cfg.instance_size, self.cfg.instance_size)
        instances_imgs = [center_crop_and_resize(img, bbox, instance_size)
                       for bbox in self.iter_scaled_instance_bboxes()]
        instances_imgs = np.stack(instances_imgs, axis=0)
        
        instances_imgs_tensor = cv_img_to_tensor(instances_imgs, self.device)
        instances_features = self.model.extract_visual_features(
            instances_imgs_tensor)
        
        responses = self.model.calc_response_map(
            self.exemplar_emb, instances_features)
        # Remove the channel dimension, as it is just 1.
        responses = responses.squeeze(1).cpu().numpy()
        # Resize response maps.
        response_size = (self.cfg.response_size, self.cfg.response_size)
        responses = np.stack(
            [cv.resize(r, response_size, interpolation=cv.INTER_CUBIC)
             for r in responses])
        
        peak_scale_pos = np.argmax(np.amax(responses, axis=(1, 2)))
        response = responses[peak_scale_pos]
        
        response -= response.min()
        response /= response.sum() + 1e-16
        response = (1 - self.cfg.cosine_win_influence) * response +\
                   self.cfg.cosine_win_influence * self.hanning
        
    
    def iter_scaled_instance_bboxes(self) -> Iterable[np.ndarray]:
        bbox_size = self.instance_bbox[2:]
        bbox_center = self.instance_bbox[:2] + bbox_size / 2
        for scale in self.search_scales:
            yield bbox_from_center_and_size(bbox_center, bbox_size * scale)

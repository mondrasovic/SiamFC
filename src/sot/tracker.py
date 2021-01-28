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
            weight_decay=self.cfg.weight_decay, momentum=self.cfg.momentum)
        self.criterion = None
        
        self.lr_scheduler = self.create_exponential_lr_scheduler(
            self.optimizer, self.cfg.initial_lr, self.cfg.ultimate_lr,
            self.cfg.n_epochs)
        
        self.response_size_upscaled: int =\
            self.cfg.response_size * self.cfg.response_upscale
        self.cosine_win: np.ndarray = self.create_square_cosine_window(
            self.response_size_upscaled)
        
        # Search for the object over multiple different scales
        # (smaller and bigger).
        self.search_scales: np.ndarray = self.create_search_scales(
            self.cfg.scale_step, self.cfg.n_scales)

        self.curr_instance_side_size: int = self.cfg.instance_size
        
        self.target_bbox = None
        self.exemplar_emb = None
    
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
        
        self.target_bbox = BBox(*bbox)
        self.curr_instance_side_size = calc_bbox_side_size_with_context(
            self.target_bbox)
        size_ratio = self.cfg.exemplar_size / self.cfg.instance_size
        exemplar_side_size = self.curr_instance_side_size * size_ratio
        
        exemplar_bbox = BBox.build_from_center_and_size(
            self.target_bbox.center,
            np.asarray((exemplar_side_size, exemplar_side_size)))
        exemplar_img = center_crop_and_resize(
            img, exemplar_bbox,
            (self.cfg.exemplar_size, self.cfg.exemplar_size))
        
        exemplar_img_tensor = cv_img_to_tensor(exemplar_img, self.device)
        self.exemplar_emb: torch.Tensor = self.model.extract_visual_features(
            exemplar_img_tensor)
    
    @torch.no_grad()
    def update(self, img: np.ndarray) -> np.ndarray:
        assert img.ndim == 3
        
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        instance_size = (self.cfg.instance_size, self.cfg.instance_size)
        instances_imgs = [
            center_crop_and_resize(img, bbox, instance_size)
            for bbox in self.iter_target_centered_scaled_instance_bboxes()]
        instances_imgs = np.stack(instances_imgs, axis=0)
        
        instances_imgs_tensor = cv_img_to_tensor(instances_imgs, self.device)
        self.model.eval()
        instances_features = self.model.extract_visual_features(
            instances_imgs_tensor)
        
        responses = self.model.calc_response_map(
            self.exemplar_emb, instances_features)
        # Remove the channel dimension, as it is just 1.
        responses = responses.squeeze(1).cpu().numpy()
        
        response_size_upscaled = (
            self.response_size_upscaled, self.response_size_upscaled)
        responses = np.stack(
            [cv.resize(r, response_size_upscaled, interpolation=cv.INTER_CUBIC)
             for r in responses])
        
        peak_scale_pos = np.argmax(np.amax(responses, axis=(1, 2)))
        peak_scale = self.search_scales[peak_scale_pos]
        
        response = responses[peak_scale_pos]
        response -= response.min()
        response /= response.sum() + 1e-16
        response = (1 - self.cfg.cosine_win_influence) * response +\
                   self.cfg.cosine_win_influence * self.cosine_win
        
        # The assumption is that the peak response value is in the center of the
        # response map. Thus, we compute the change with respect to the center
        # and convert it back to the pixel coordinates in the image.
        peak_response_pos = np.unravel_index(response.argmax(), response.shape)
        disp_in_response = peak_response_pos - self.response_size_upscaled // 2
        disp_in_instance = disp_in_response *\
                           (self.cfg.total_stride / self.cfg.response_upscale)
        disp_in_image = disp_in_instance * self.target_bbox.center *\
                        (peak_scale / self.cfg.instance_size)
        self.target_bbox.shift(disp_in_image)
        
        # Update target scale.
        new_scale = (1 - self.cfg.scale_damping) * 1.0 +\
                    (self.cfg.scale_damping * peak_scale)
        self.curr_instance_side_size *= new_scale
        self.target_bbox.rescale(new_scale, new_scale)
        
        return self.target_bbox.as_xywh()
    
    def iter_target_centered_scaled_instance_bboxes(self) -> Iterable[BBox]:
        size = np.asarray(
            self.curr_instance_side_size, self.curr_instance_side_size)
        bbox = BBox.build_from_center_and_size(self.target_bbox.center, size)
        
        for scale in self.search_scales:
            yield bbox.rescale(scale, scale, in_place=False)
    
    @staticmethod
    def create_square_cosine_window(size: int) -> np.ndarray:
        assert size > 0

        # Create a normalized cosine (Hanning) window.
        hanning_1d = np.hanning(size)
        hanning_2d = np.outer(hanning_1d, hanning_1d)
        hanning_2d /= np.sum(hanning_2d)
    
        return hanning_2d
    
    @staticmethod
    def create_search_scales(scale_step: float, count: int) -> np.ndarray:
        assert count > 0

        n_half_search_scales = count // 2
        search_scales = scale_step ** np.linspace(
            -n_half_search_scales, n_half_search_scales, count)
        
        return search_scales
    
    @staticmethod
    def create_exponential_lr_scheduler(
            optimizer, initial_lr: float, ultimate_lr: float,
            n_epochs: int) -> optim.lr_scheduler.ExponentialLR:
        assert n_epochs > 0
        
        # Learning rate is geometrically annealed at each epoch. Starting from
        # A and terminating at B, then the known gamma factor x for n epochs
        # is computed as
        #         A * x ^ n = B,
        #                 x = (B / A)^(1 / n).
        gamma = np.power(ultimate_lr / initial_lr, 1.0 / n_epochs)
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma)
        
        return lr_scheduler

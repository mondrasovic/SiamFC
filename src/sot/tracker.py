#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Milan Ondrasovic <milan.ondrasovic@gmail.com>

from typing import Iterable, Optional, Union

import cv2 as cv
import numpy as np
import torch
from got10k.trackers import Tracker

from sot.bbox import BBox
from sot.cfg import TrackerConfig
from sot.model import SiamFCModel
from sot.utils import (
    assure_int_bbox, calc_bbox_side_size_with_context, center_crop_and_resize,
    ImageT, pil_to_tensor,
)


class TrackerSiamFC(Tracker):
    def __init__(
            self, cfg: TrackerConfig, device: Union[torch.device, str],
            model_path: Optional[str] = None) -> None:
        super().__init__(name='SiamFC', is_deterministic=True)
        
        self.cfg: TrackerConfig = cfg
        
        if isinstance(device, torch.device):
            self.device: torch.device = device
        else:
            self.device: torch.device = torch.device(device)
        
        self.model: SiamFCModel = SiamFCModel(self.cfg.response_map_scale)
        if model_path is not None:
            self.model.load_state_dict(
                torch.load(
                    model_path, map_location=lambda storage, location: storage))
        self.model = self.model.to(self.device)
        
        self.response_size_upscaled: int = \
            self.cfg.response_size * self.cfg.response_upscale
        self.cosine_win: np.ndarray = self.create_square_cosine_window(
            self.response_size_upscaled)
        
        self.search_scales: np.ndarray = self.create_search_scales(
            self.cfg.scale_step, self.cfg.n_scales)
        
        self.curr_instance_side_size: int = self.cfg.instance_size
        
        self.target_bbox = None
        self.exemplar_emb = None
    
    @torch.no_grad()
    def init(self, img: ImageT, bbox: np.ndarray) -> None:
        assert (len(bbox) == 4) and (bbox.ndim == 1)
        
        self.model.eval()
        
        bbox = assure_int_bbox(bbox)
        self.target_bbox = BBox(*bbox)
        self.curr_instance_side_size = calc_bbox_side_size_with_context(
            self.target_bbox)
        size_ratio = self.cfg.exemplar_size / self.cfg.instance_size
        exemplar_side_size = int(round(
            self.curr_instance_side_size * size_ratio))
        
        exemplar_bbox = BBox.build_from_center_and_size(
            self.target_bbox.center,
            np.asarray((exemplar_side_size, exemplar_side_size)))
        exemplar_img = center_crop_and_resize(
            img, exemplar_bbox,
            (self.cfg.exemplar_size, self.cfg.exemplar_size))
        
        exemplar_img_tensor = torch.unsqueeze(pil_to_tensor(exemplar_img), 0)
        exemplar_img_tensor = exemplar_img_tensor.to(self.device)
        self.exemplar_emb = self.model.extract_visual_features(
            exemplar_img_tensor)
        # Copy the exemplar as many times as we have scale factors. Since we
        # employ the trick with grouped convolutions, the embedding vector needs
        # to be present for each group.
        self.exemplar_emb = self.exemplar_emb.repeat(self.cfg.n_scales, 1, 1, 1)
    
    @torch.no_grad()
    def update(self, img: ImageT) -> np.ndarray:
        self.model.eval()
        
        # Search for the object over multiple different scales
        # (smaller and bigger).
        instance_size = (self.cfg.instance_size, self.cfg.instance_size)
        instances_imgs = (
            center_crop_and_resize(img, bbox, instance_size)
            for bbox in self.iter_target_centered_scaled_instance_bboxes())
        
        instances_imgs_tensor = torch.stack(
            [pil_to_tensor(img) for img in instances_imgs])
        instances_imgs_tensor = instances_imgs_tensor.to(self.device)
        instances_features = self.model.extract_visual_features(
            instances_imgs_tensor)
        
        responses = self.model.calc_response_map(
            self.exemplar_emb, instances_features)
        # Remove the channel dimension, as it is just 1.
        responses = responses.squeeze(1).cpu().numpy()
        
        # Increase response maps size.
        response_size_upscaled = (
            self.response_size_upscaled, self.response_size_upscaled)
        responses = np.stack(
            [cv.resize(r, response_size_upscaled, interpolation=cv.INTER_CUBIC)
             for r in responses])
        
        # Penalize scales.
        responses[:self.cfg.n_scales // 2] *= self.cfg.scale_penalty
        responses[self.cfg.n_scales // 2 + 1:] *= self.cfg.scale_penalty
        
        peak_scale_pos = np.argmax(np.amax(responses, axis=(1, 2)))
        peak_scale = self.search_scales[peak_scale_pos]
        
        # Normalize response map so that it sums to one.
        response = responses[peak_scale_pos]
        response -= response.min()
        response /= response.sum() + 1.e-16
        
        response = (1 - self.cfg.cosine_win_influence) * response + \
                   self.cfg.cosine_win_influence * self.cosine_win
        
        # The assumption is that the peak response value is in the center of the
        # response map. Thus, we compute the change with respect to the center
        # and convert it back to the pixel coordinates in the image.
        peak_response_pos = np.asarray(
            np.unravel_index(response.argmax(), response.shape))
        
        disp_in_response = peak_response_pos - self.response_size_upscaled // 2
        disp_in_instance = disp_in_response * \
                           (self.cfg.total_stride / self.cfg.response_upscale)
        disp_in_image = disp_in_instance * self.curr_instance_side_size * \
                        (peak_scale / self.cfg.instance_size)
        disp_in_image = disp_in_image.round().astype(np.int)
        
        # Update target scale.
        new_scale = (1 - self.cfg.scale_damping) * 1.0 + \
                    (self.cfg.scale_damping * peak_scale)
        self.curr_instance_side_size *= new_scale
        
        # Change from [row, col] to [x, y] coordinates.
        self.target_bbox.shift(disp_in_image[::-1])
        self.target_bbox.rescale(new_scale, new_scale)
        
        return self.target_bbox.as_xywh()
    
    def iter_target_centered_scaled_instance_bboxes(self) -> Iterable[BBox]:
        side_size = int(round(self.curr_instance_side_size))
        size = np.asarray((side_size, side_size))
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

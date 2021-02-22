#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Milan Ondrasovic <milan.ondrasovic@gmail.com>

import math
from typing import Tuple, Optional

import cv2 as cv
import numpy as np

from sot.utils import SizeT

ColorT = Tuple[int, int, int]


def render_response_map(
        response: np.ndarray, size: Optional[SizeT] = None) -> np.ndarray:
    response = ((response / response.max()) * 255).round().astype(np.uint8)
    if size is not None:
        response = cv.resize(response, size, interpolation=cv.INTER_CUBIC)
    response = cv.applyColorMap(response, cv.COLORMAP_JET)
    return response


def concat_imgs(
        imgs, *, row: bool = True,
        border_value: ColorT = (0, 0, 0)) -> np.ndarray:
    if row:
        dim_idx = 0  # Height
        pad_indexes = (0, 1)  # Top, bottom
        stack_func = np.hstack
    else:
        dim_idx = 1  # Width
        pad_indexes = (2, 3)  # Left, right
        stack_func = np.vstack

    max_dim = max(img.shape[dim_idx] for img in imgs)
    padded_imgs = []

    for img in imgs:
        dim_diff_half = (max_dim - img.shape[dim_idx]) / 2
        pads = [0, 0, 0, 0]
        pads[pad_indexes[0]] = int(math.floor(dim_diff_half))
        pads[pad_indexes[1]] = int(math.ceil(dim_diff_half))

        padded_img = cv.copyMakeBorder(
            img, *pads, borderType=cv.BORDER_CONSTANT, value=border_value)
        padded_imgs.append(padded_img)

    return stack_func(padded_imgs)


class SiameseTrackingVisualizer:
    def __init__(
            self, exemplar_img: np.ndarray, *,
            border_value: ColorT = (0, 0, 0),
            win_name: str = "Siamese Tracking Preview",
            wait_key: int = 0, quit_key: str = 'q') -> None:
        self.exemplar_img: np.ndarray = exemplar_img
        self.border_value: ColorT = border_value
        self.win_name: str = win_name
        self.wait_key: int = wait_key
        self.quit_key: str = quit_key

    def show_curr_state(
            self, curr_frame: np.ndarray, instance_img: np.ndarray,
            response_map: np.ndarray, bbox_pred: np.ndarray) -> bool:
        pt1 = tuple(bbox_pred[:2])
        pt2 = tuple(bbox_pred[:2] + bbox_pred[2:])
        cv.rectangle(curr_frame, pt1, pt2, (0, 255, 0), 3, cv.LINE_AA)
        
        response_map = render_response_map(response_map)
        row = concat_imgs(
            (self.exemplar_img, instance_img, response_map), row=True,
            border_value=self.border_value)
        preview_img = concat_imgs(
            (curr_frame, row), row=False, border_value=self.border_value)
        cv.imshow(self.win_name, preview_img)
        key = cv.waitKey(self.wait_key) & 0xff
        ret = (key != ord(self.quit_key))
        
        return ret
    
    def close(self) -> None:
        cv.destroyWindow(self.win_name)

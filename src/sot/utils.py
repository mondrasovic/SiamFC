from typing import Optional, Tuple, Union

import cv2 as cv
import numpy as np
import torch
from PIL import Image

from sot.bbox import BBox


Size = Union[np.ndarray, Tuple[int, int]]


def calc_bbox_side_size_with_context(bbox: BBox) -> float:
    context_size = bbox.size.mean()  # Average dimension.
    scaled_side_size = np.sqrt(np.prod(bbox.size + context_size))
    return scaled_side_size


def center_crop_and_resize(
        img: np.ndarray, bbox: BBox, target_size: Size,
        border: Optional[Union[int, Tuple[int, ...]]] = None,
        interpolation=cv.INTER_CUBIC) -> np.ndarray:
    assert img.ndim == 3, "expected three dimensional image"
    
    # Size as width and height
    bbox_corners = bbox.as_corners()
    img_size = (img.shape[1], img.shape[0])
    paddings = np.concatenate((-bbox_corners[:2], bbox_corners[2:] - img_size))
    max_padding = np.maximum(paddings, 0).max()
    
    if max_padding > 0:
        if border is None:
            border = tuple(int(c) for c in np.mean(img, axis=(0, 1)).round())
        
        img = cv.copyMakeBorder(
            img, max_padding, max_padding, max_padding, max_padding,
            borderType=cv.BORDER_CONSTANT, value=border)
        bbox_corners += max_padding
    
    patch = img[
            bbox_corners[1]:bbox_corners[3], bbox_corners[0]:bbox_corners[2]]
    patch = cv.resize(patch, dsize=target_size, interpolation=interpolation)
    
    return patch


def create_ground_truth_mask_and_weight(
        size: Size, radius: float, total_stride: int,
        batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    assert radius > 0, "radius must be positive"
    assert total_stride > 0, "total stride must be positive"
    assert batch_size > 0, "batch size must be positive"

    width, height = size

    xs = np.arange(width, dtype=np.float32) - (width - 1) / 2
    ys = np.arange(height, dtype=np.float32) - (height - 1) / 2
    XS, YS = np.meshgrid(xs, ys)

    dist_matrix = np.sqrt(XS ** 2 + YS ** 2)
    mask_mat = np.zeros((height, width))
    mask_mat[dist_matrix <= radius / total_stride] = 1
    mask_mat = mask_mat[None, ...]  # Add channel dimension.

    weight_mat = np.empty_like(mask_mat)
    weight_mat[mask_mat == 1] = 0.5 / np.sum(mask_mat == 1)
    weight_mat[mask_mat == 0] = 0.5 / np.sum(mask_mat == 0)

    mask_mat = mask_mat[None, ...]  # Add batch dimension.
    mask_mat = np.repeat(mask_mat, batch_size, axis=0)

    mask_mat = mask_mat.astype(np.float32)
    weight_mat = weight_mat.astype(np.float32)

    return mask_mat, weight_mat


def cv_to_pil_img(img: np.ndarray) -> Image:
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return Image.fromarray(img)


def pil_to_cv_img(img: Image) -> np.ndarray:
    return cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)


def cv_img_to_tensor(
        img: np.ndarray, device: Optional[torch.device] = None) -> torch.Tensor:
    assert 3 <= img.ndim <= 4, "expected image with 3 or 4 dimensions"
    
    tensor = torch.from_numpy(img)
    
    if device is not None:
        tensor = tensor.to(device)
    
    if tensor.ndim == 3:  # A single image (height, width, channels).
        tensor = tensor.unsqueeze(0)  # Add batch dimension.
    
    # Swap channels to get [batch_size, channels, height, width].
    return tensor.permute(0, 3, 1, 2).float()

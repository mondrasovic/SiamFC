import numbers
from typing import Optional, Tuple, Union

import cv2 as cv
import numpy as np
import torch
from PIL import Image, ImageOps, ImageStat
from torchvision import transforms

from sot.bbox import BBox


Size = Union[np.ndarray, Tuple[int, int]]
ImageT = Image.Image


def calc_bbox_side_size_with_context(bbox: BBox) -> float:
    # Exemplar and instance (search) sizes.
    # The endeavor is to resize the image so that the bounding box plus the
    # margin have a fixed area. In the original paper, the constraint was
    #         s(w + 2p)s(h + 2p) = A,
    # where p = (w + h) / 4, in other words, half of the average dimension,
    # and A = 127^2. However, it can be implemented as
    #         s = sqrt(A / ((w + p)(h + p)),
    # given p = (w + h). The multiplication by 2 essentially cancels out
    # the "half" in p.
    
    context_size = bbox.size.mean()  # Average dimension.
    scaled_side_size = np.sqrt(np.prod(bbox.size + context_size))
    return scaled_side_size


def center_crop_and_resize(
        img: ImageT, bbox: BBox, target_size: Size,
        border: Optional[Union[int, Tuple[int, ...]]] = None,
        interpolation=Image.BICUBIC) -> ImageT:
    bbox_corners = bbox.as_corners()
    paddings = np.concatenate((-bbox_corners[:2], bbox_corners[2:] - img.size))
    max_padding = np.maximum(paddings, 0).max()
    
    if max_padding > 0:
        if border is None:
            border = tuple(int(round(c)) for c in ImageStat.Stat(img).mean)
        
        img = ImageOps.expand(img, border=max_padding, fill=border)
        bbox_corners += max_padding
    
    bbox_corners = tuple((bbox_corners).astype(int))
    patch = img.crop(bbox_corners)
    patch = patch.resize(target_size, interpolation)
    
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
    n_positives = np.sum(mask_mat == 1)
    n_negatives = np.sum(mask_mat == 0)
    weight_mat[mask_mat == 1] = 0.5 / n_positives
    weight_mat[mask_mat == 0] = 0.5 / n_negatives
    
    mask_mat = mask_mat[None, ...]  # Add batch dimension.
    mask_mat = np.repeat(mask_mat, batch_size, axis=0)
    
    mask_mat = mask_mat.astype(np.float32)
    weight_mat = weight_mat.astype(np.float32)
    
    return mask_mat, weight_mat


def cv_show_tensor_as_img(img: torch.Tensor, win_name: str) -> None:
    img = img.cpu().detach().squeeze(0).numpy()
    img = np.transpose(img, axes=(1, 2, 0))
    cv.imshow(win_name, img)


def cv_wait_key_and_destroy_all(delay: int = 0, quit_key: str = 'q') -> bool:
    key = cv.waitKey(delay) & 0xff
    cv.destroyAllWindows()
    return key == ord(quit_key)


def show_response_maps(
        responses: np.ndarray, size: Size = (408, 408),
        wait_key: int = 0) -> None:
    for i, response in enumerate(responses, start=1):
        response = ((response / response.max()) * 255).round().astype(np.uint8)
        response = cv.resize(response, size, interpolation=cv.INTER_CUBIC)
        response = cv.applyColorMap(response, cv.COLORMAP_JET)
        cv.imshow(f"{i:04d} response map", response)
    
    cv.waitKey(wait_key)
    cv.destroyAllWindows()


def cv_to_pil_img(img: np.ndarray) -> Image:
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return Image.fromarray(img)


def pil_to_cv_img(img: Image) -> np.ndarray:
    return cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)


_pil_to_tensor_transform = transforms.PILToTensor()


def pil_to_tensor(img: ImageT) -> torch.Tensor:
    return _pil_to_tensor_transform(img).float()


def assure_int_bbox(bbox: np.ndarray) -> np.ndarray:
    if issubclass(bbox.dtype.type, numbers.Integral):
        return bbox
    else:
        return bbox.round().astype(np.int)

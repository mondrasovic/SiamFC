import numbers

from typing import Optional, Tuple, Union

import cv2 as cv
import numpy as np
import torch

from PIL import Image

from sot.bbox import BBox


Size = Union[np.ndarray, Tuple[int, int]]
ImageT = Union[np.ndarray, Image.Image]


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


def assure_numpy_img(img: ImageT) -> np.ndarray:
    if isinstance(img, np.ndarray):
        return img
    elif isinstance(img, Image.Image):
        return pil_to_cv_img(img)
    else:
        raise ValueError("unsupported image type")


def assure_int_bbox(bbox: np.ndarray) -> np.ndarray:
    if issubclass(bbox.dtype.type, numbers.Integral):
        return bbox
    else:
        return bbox.round().astype(np.int)

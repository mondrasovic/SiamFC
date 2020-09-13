import dataclasses
from typing import Tuple, Union, Callable, Optional
from numbers import Number

import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import Compose, CenterCrop, RandomCrop, ToTensor
from PIL import Image, ImageStat, ImageOps

BBoxT = Union[Tuple[Number, Number, Number, Number], np.ndarray]
TransformsT = Optional[Union[Callable, Compose]]

@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    pairs_per_sequence: int = 10
    exemplar_size: int = 127
    instance_size: int = 255
    max_pair_distance: int = 100  # Distance in the no. of frames between exemplar and instance.
    context: float = 0.5
    min_bbox_area: int = 10


class PairwiseDataset(Dataset):
    def __init__(
            self,
            data_sequence: Dataset,
            config: DatasetConfig,
            transforms: TransformsT = None) -> None:
        super().__init__()
        
        self._config: DatasetConfig = config
        self._data_sequence: Dataset = data_sequence
        self._transforms: TransformsT = transforms
        
        self._indices: np.ndarray = np.random.permutation(len(self._data_sequence))
    
    def __getitem__(self, index: int) -> Tuple[Image, Image]:
        assert index >= 0
        
        index = self._indices[index % len(self._data_sequence)]
        image_files, annotations = self._data_sequence[index]
        
        # Remove too small objects.
        valid_indices = annotations[:, 2:].prod(axis=1) >= self._config.min_bbox_area
        valid_image_files = np.array(image_files)[valid_indices]
        valid_annotations = annotations[valid_indices, :]
        
        exemplar_index, instance_index = self._sample_pair_indices(len(valid_image_files))
        
        exemplar_image = self._read_image_and_transform(
            valid_image_files[exemplar_index],
            valid_annotations[exemplar_index])
        if exemplar_index != instance_index:
            instance_image = self._read_image_and_transform(
                valid_image_files[instance_index],
                valid_annotations[instance_index])
        else:
            instance_image = exemplar_image.copy()
        
        return exemplar_image, instance_image
    
    def __len__(self) -> int:
        return len(self._data_sequence) * self._config.pairs_per_sequence
    
    def _sample_pair_indices(self, count: int) -> Tuple[int, int]:
        assert count > 0
        
        max_distance = min(count - 1, self._config.max_pair_distance)
        rand_indices = np.random.choice(max_distance + 1, 2)
        rand_start = np.random.randint(count - max_distance)
        
        return rand_indices + rand_start
    
    def _read_image_and_transform(self, image_path: str, bbox: BBoxT) -> Image:
        image = Image.open(image_path)
        image = self._crop_and_resize(image, bbox)
        image = 255 * self._transforms(image)
        return image
    
    def _crop_and_resize(self, image: Image, bbox: BBoxT) -> Image:
        # Convert the exemplar (target) bounding box to 0-indexed and center-based.
        bbox_converted = np.float32(((bbox[0] - 1) + (bbox[2] - 1) / 2.0,
                                     (bbox[1] - 1) + (bbox[3] - 1) / 2.0,
                                     *bbox[2:]))
        center, target_size = bbox_converted[:2], bbox_converted[2:]

        # TODO Refactor this.
        # Exemplar and instance (search) sizes.
        # The endeavor is to resize the image so that the bounding box plus the margin have a fixed
        # area. In the original paper, the constraint was
        #         s(w + 2p)s(h + 2p) = A,
        # where p = (w + h) / 4, in other words, half of the average dimension, and A = 127^2.
        # However, it can be implemented as
        #         s = sqrt(A / ((w + p)(h + p)),
        # given p = (w + h). The multiplication by 2 essentially cancels out the "half" in p.
        context_size = self._config.context * np.sum(target_size)  # Average dimension.
        exemplar_with_context_size = np.sqrt(np.prod(target_size + context_size))
        scale = exemplar_with_context_size / self._config.exemplar_size
        instance_size_adjusted = round(self._config.instance_size * scale)
        
        # Convert the bounding box to a 0-indexed, corner-based representation.
        # Images are centered on the target.
        half_size = (instance_size_adjusted - 1) / 2.0
        bbox_corners = np.round(np.concatenate((
            np.round(center - half_size),
            np.round(center - half_size) + instance_size_adjusted
        ))).astype(int)
        
        # Pad the image if necessary. It computes the number of pixels to add (subtract) to each
        # corners so that the image patch is within the image region.
        corner_paddings = np.concatenate((-bbox_corners[:2], bbox_corners[2:] - image.size))
        max_padding = max(0, int(np.max(corner_paddings)))
        
        if max_padding > 0:
            # The PIL library does not support a float RGB image.
            avg_color = tuple(map(lambda c: int(round(c)), ImageStat.Stat(image).mean))
            image = ImageOps.expand(image, border=max_padding, fill=avg_color)
        
        # Crop the image patch. Compute a 4-tuple defining the left, upper, right, and lower pixel
        # coordinate.
        corners = tuple((bbox_corners + max_padding).astype(int))
        patch = image.crop(corners)
        
        # Resize to the required search (instance) size.
        out_size = (self._config.instance_size, self._config.instance_size)
        patch = patch.resize(out_size, Image.BILINEAR)
        
        return patch

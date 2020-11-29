import numpy as np

from typing import Tuple, Union, Callable, Optional, Sequence, cast

from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor
from PIL import Image, ImageStat, ImageOps

from common import TrackerConfig, BBoxT, TensorT

TransformsT = Optional[Union[Callable, Compose]]


class PairwiseDataset(Dataset):
    def __init__(self, data_seq: Sequence, config: TrackerConfig) -> None:
        super().__init__()
        
        self.config: TrackerConfig = config
        self.data_seq: Sequence = data_seq
        
        self.indices: np.ndarray = np.random.permutation(len(self.data_seq))
        
        # TODO Add data augmentation.
        self.transform_z = Compose([ToTensor()])
        self.transform_x = Compose([ToTensor()])
    
    def __getitem__(self, index: int) -> Tuple[TensorT, TensorT]:
        assert index >= 0
        
        index = self.indices[index % len(self.data_seq)]
        img_files, annos = self.data_seq[index]
        
        # Remove too small objects.
        valid_indices = annos[:, 2:].prod(axis=1) >= self.config.min_bbox_area
        valid_img_files = np.array(img_files)[valid_indices]
        valid_annos = annos[valid_indices, :]
        
        z_index, x_index = self._sample_pair_indices(len(valid_img_files))
        z_img = self._read_image_and_transform(
            valid_img_files[z_index], valid_annos[z_index], self.transform_z)
        if z_index != x_index:
            x_img = self._read_image_and_transform(
                valid_img_files[x_index], valid_annos[x_index],
                self.transform_x)
        else:
            x_img = z_img.copy()
        
        return z_img, x_img
    
    def __len__(self) -> int:
        return len(self.data_seq) * self.config.pairs_per_seq
    
    def _sample_pair_indices(self, count: int) -> Tuple[int, int]:
        assert count > 0
        
        max_distance = min(count - 1, self.config.max_pair_dist)
        rand_indices = np.random.choice(max_distance + 1, 2)
        rand_start = np.random.randint(count - max_distance)
        return rand_indices + rand_start
    
    def _read_image_and_transform(
            self, img_path: str, bbox: BBoxT, transform: Callable) -> Image:
        img = Image.open(img_path)
        img = self._crop_and_resize(img, bbox)
        img = 255 * transform(img)
        return img
    
    def _crop_and_resize(self, img: Image, bbox: BBoxT) -> Image:
        # Convert the exemplar (target) bounding box to 0-indexed and
        # center-based.
        bbox_converted = np.float32(
            ((bbox[0] - 1) + (bbox[2] - 1) / 2.0,
             (bbox[1] - 1) + (bbox[3] - 1) / 2.0, *bbox[2:]))
        center, target_size = bbox_converted[:2], bbox_converted[2:]

        # TODO Refactor this.
        # Exemplar and instance (search) sizes.
        # The endeavor is to resize the image so that the bounding box plus the
        # margin have a fixed area. In the original paper, the constraint was
        #         s(w + 2p)s(h + 2p) = A,
        # where p = (w + h) / 4, in other words, half of the average dimension,
        # and A = 127^2. However, it can be implemented as
        #         s = sqrt(A / ((w + p)(h + p)),
        # given p = (w + h). The multiplication by 2 essentially cancels out
        # the "half" in p.
        context_size = (self.config.context *
                        np.sum(target_size))  # Average dimension.
        z_with_context_size = np.sqrt(np.prod(target_size + context_size))
        scale = z_with_context_size / self.config.exemplar_size
        x_size_adjusted = round(self.config.instance_size * scale)
        
        # Convert the bounding box to a 0-indexed, corner-based representation.
        # Images are centered on the target.
        half_size = (x_size_adjusted - 1) / 2.0
        bbox_corners = np.round(np.concatenate((
            np.round(center - half_size),
            np.round(center - half_size) + x_size_adjusted))).astype(int)
        
        # Pad the image if necessary. It computes the number of pixels to add
        # (subtract) to each corners so that the image patch is within the image
        # region.
        corner_paddings = np.concatenate(
            (-bbox_corners[:2], bbox_corners[2:] - img.size))
        max_padding = max(0, int(np.max(corner_paddings)))
        
        if max_padding > 0:
            # The PIL library does not support a float RGB image.
            avg_color = tuple((int(round(c)) for c in ImageStat.Stat(img).mean))
            img = ImageOps.expand(img, border=max_padding, fill=avg_color)
        
        # Crop the image patch. Compute a 4-tuple defining the left, upper,
        # right, and lower pixel coordinate.
        corners = tuple((bbox_corners + max_padding).astype(int))
        patch = img.crop(corners)
        
        # Resize to the required search (instance) size.
        out_size = (self.config.instance_size, self.config.instance_size)
        patch = patch.resize(out_size, Image.BILINEAR)  # TODO Use BICUBIC.
        
        return patch


if __name__ == '__main__':
    import os
    import torch
    import glob
    from got10k.datasets import ImageNetVID
    import shutil
    
    # shutil.rmtree('./cache', ignore_errors=True)
    dataset_dir = '../../../datasets/ILSVRC2015_VID'
    d = r'..\..\..\datasets\ILSVRC2015_VID\Annotations\VID\train\VID\train\ILSVRC2015_VID_train_0000\ILSVRC2015_train_00000000'
    d = r'..\..\..\datasets\ILSVRC2015_VID\Annotations\VID\train'
    seq_dirs_ = sorted(glob.glob(os.path.join(
        dataset_dir, 'Data/VID/val/ILSVRC2015_val_*')))
    for s in seq_dirs_:
        print(s.split('/')[-2:])
    
    seq_dataset = ImageNetVID(dataset_dir, subset=('train', 'val'))
    config = TrackerConfig()
    pair_dataset = PairwiseDataset(cast(Sequence, seq_dataset), config)
    # cuda = torch.cuda.is_available()
    # pair_loader = torch.utils.data.DataLoader(
    #     pair_dataset, batch_size=config.batch_size, shuffle=True,
    #     pin_memory=cuda, drop_last=True, num_workers=4)
    pair_loader = torch.utils.data.DataLoader(
        pair_dataset, batch_size=config.batch_size)
    for i, batch in enumerate(pair_loader):
        if i > 3:
            break

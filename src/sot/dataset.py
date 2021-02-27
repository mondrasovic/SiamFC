#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Milan Ondrasovic <milan.ondrasovic@gmail.com>

import abc
import collections
import dataclasses
import os
import pathlib
import xml.etree.ElementTree as ET
from typing import (
    Callable, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple,
    Union,
)

import cv2 as cv
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from sot.bbox import BBox
from sot.cfg import TrackerConfig
from sot.utils import (
    calc_bbox_side_size_with_context, center_crop_and_resize, rand_uniform,
    ImageT
)


class RandomStretch:
    INTERPOLATIONS = (
        Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS, Image.BOX,
        Image.HAMMING)
    
    def __init__(self, stretch_coef: float) -> None:
        self.stretch_coef: float = stretch_coef
    
    def __call__(self, image: ImageT) -> ImageT:
        new_scale = 1.0 + rand_uniform(-self.stretch_coef, self.stretch_coef)
        new_size = np.round(np.array(image.size, float) * new_scale).astype(int)
        interpolation = np.random.choice(self.INTERPOLATIONS)
        resized_image = image.resize(new_size, interpolation)
        
        return resized_image


@dataclasses.dataclass(frozen=True)
class TrackData:
    img_file_path: str
    bbox: Optional[BBox]


class TrackingDataset:
    def __init__(self, name: str, root_dir_path: str) -> None:
        self.name: str = name
        self.root_dir_path: str = root_dir_path
        
        self.track_ids: Sequence[str] = []
    
    def __getitem__(self, index: int) -> Tuple[Sequence[str], np.ndarray]:
        track_id = self.track_ids[index]
        img_file_paths = tuple(self.read_img_file_paths(track_id))
        annos = self.read_annotations(track_id)
        return img_file_paths, annos
    
    def __len__(self) -> int:
        return len(self.track_ids)
    
    def initialize(self):
        self.track_ids = tuple(self.read_track_ids())
    
    @abc.abstractmethod
    def read_track_ids(self) -> Iterable[str]:
        pass
    
    @abc.abstractmethod
    def read_img_file_paths(self, track_id: str) -> Iterable[str]:
        pass
    
    @abc.abstractmethod
    def read_annotations(self, track_id) -> np.ndarray:
        pass


class ImageNetVideoDataset(TrackingDataset):
    def __init__(
            self, root_dir_path: str,
            subsets: Union[str, Sequence[str]] = 'train'):
        super().__init__('ImageNetVID', root_dir_path)
        
        if isinstance(subsets, str):
            subsets = (subsets,)
        assert all(subset in ('train', 'val', 'test') for subset in subsets)
        
        self.subsets: Sequence[str] = subsets
        self.track_data: DefaultDict[str, List[TrackData]] = \
            collections.defaultdict(list)
    
    def read_track_ids(self) -> Iterable[str]:
        # In this dataset, one video may contain multiple tracks. Even though
        # a track ID is available in the XML files, it is not globally unique,
        # only within a particular sequence.
        root_dir = pathlib.Path(self.root_dir_path)
        
        for subset in self.subsets:
            if subset in ('train', 'val'):
                data_subset_dir = root_dir / 'Data' / 'VID' / subset
                anno_subset_dir = root_dir / 'Annotations' / 'VID' / subset
                
                for seq_dir in anno_subset_dir.rglob(f"ILSVRC2015_{subset}_*"):
                    annos_xml_files = [
                        tuple(self._read_annos_from_xml(str(anno_xml_file)))
                        for anno_xml_file in seq_dir.glob('*.xml')]
                    track_bbox_counter = collections.Counter(
                        track_id
                        for xml_content in annos_xml_files
                        for track_id, *_ in xml_content)
                    
                    for xml_content in annos_xml_files:
                        for track_id, img_file_path, bbox in xml_content:
                            if track_bbox_counter[track_id] < 2:
                                continue
                            img_file_path_ex = data_subset_dir / img_file_path
                            track_id_ex = f'{seq_dir.name}_{track_id}'
                            track_data = TrackData(str(img_file_path_ex), bbox)
                            self.track_data[track_id_ex].append(track_data)
            else:  # 'test'
                test_data_dir = root_dir / 'Data' / 'VID' / 'test'
                for track_id, img_file_path in \
                        self._read_test_track_ids_and_img_paths(test_data_dir):
                    track_data = TrackData(img_file_path, None)
                    self.track_data[track_id].append(track_data)
        return self.track_data.keys()
    
    def read_img_file_paths(self, track_id: str) -> Iterable[str]:
        return (data.img_file_path for data in self.track_data[track_id])
    
    def read_annotations(self, track_id) -> np.ndarray:
        annos = [data.bbox.as_xywh() for data in self.track_data[track_id]]
        return np.array(annos, dtype=np.int)
    
    @staticmethod
    def _read_annos_from_xml(
            xml_file_path: str) -> Iterable[Tuple[str, str, BBox]]:
        root = ET.ElementTree(file=xml_file_path).getroot()
        folder = root.find('folder').text
        filename = root.find('filename').text
        img_file_path = os.path.join(folder, f"{filename}.JPEG")
        
        for object_node in root.findall('object'):
            track_id = object_node.find('trackid').text
            bbox_node = object_node.find('bndbox')
            
            x_min = int(bbox_node.find('xmin').text)
            y_min = int(bbox_node.find('ymin').text)
            x_max = int(bbox_node.find('xmax').text)
            y_max = int(bbox_node.find('ymax').text)
            
            bbox = BBox(x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)
            
            yield track_id, img_file_path, bbox
    
    @staticmethod
    def _read_test_track_ids_and_img_paths(
            test_data_dir: pathlib.Path) -> Iterable[Tuple[str, str]]:
        for track_dir in test_data_dir.iterdir():
            track_id = track_dir.name
            
            for img_file in track_dir.iterdir():
                yield track_id, str(img_file)


class OTBDataset(TrackingDataset):
    def __init__(self, root_dir_path: str) -> None:
        super().__init__('OTB2013', root_dir_path)
        
        self.track_data: Dict[str, List[TrackData]] = {}
    
    def read_track_ids(self) -> Iterable[str]:
        root_dir = pathlib.Path(self.root_dir_path)
        
        for track_dir in root_dir.iterdir():
            if not track_dir.is_dir():
                continue
            
            imgs_dir = track_dir / 'img'
            bboxes_file = track_dir / 'groundtruth_rect.txt'
            if not bboxes_file.exists():  # TODO Incorporate both tracks.
                bboxes_file = track_dir / 'groundtruth_rect.1.txt'
            
            with open(str(bboxes_file), 'rt') as bboxes_file:
                img_file_paths = (
                    str(img_file) for img_file in imgs_dir.iterdir())
                bboxes = (
                    self.create_bbox_from_str(line)
                    for line in bboxes_file.readlines())
                
                track_data_list = []
                for img_file_path, bbox in zip(img_file_paths, bboxes):
                    track_data_list.append(TrackData(img_file_path, bbox))
                
                track_id = track_dir.name
                self.track_data[track_id] = track_data_list
        
        return self.track_data.keys()
    
    def read_img_file_paths(self, track_id: str) -> Iterable[str]:
        return (data.img_file_path for data in self.track_data[track_id])
    
    def read_annotations(self, track_id) -> np.ndarray:
        annos = [data.bbox.as_xywh() for data in self.track_data[track_id]]
        return np.array(annos, dtype=np.int)
    
    @staticmethod
    def create_bbox_from_str(bbox_str: str) -> BBox:
        sep = ',' if ',' in bbox_str else '\t'
        return BBox(*tuple(map(int, bbox_str.split(sep))))


PairItemT = Tuple[torch.Tensor, torch.Tensor]


class SiamesePairwiseDataset(Dataset):
    def __init__(self, data_seq: Sequence, cfg: TrackerConfig) -> None:
        super().__init__()
        
        self.data_seq: Sequence = data_seq
        self.cfg: TrackerConfig = cfg
        
        self.indices: np.ndarray = np.random.permutation(len(self.data_seq))
        
        self.transform_exemplar = self._build_transforms(self.cfg.exemplar_size)
        self.transform_instance = self._build_transforms(self.cfg.instance_size)
    
    def __getitem__(self, index: int) -> PairItemT:
        index = self.indices[index % len(self.data_seq)]
        img_files, annos = self.data_seq[index]
        annos = annos.astype(np.int)
        
        valid_indices = self._filter_valid_indices(annos)
        valid_img_files = np.asarray(img_files)[valid_indices]
        valid_annos = annos[valid_indices, :]
        
        n_imgs = len(valid_img_files)
        exemplar_idx, instance_idx = self._sample_pair_indices(n_imgs)
        
        exemplar_img_path = valid_img_files[exemplar_idx]
        exemplar_anno = valid_annos[exemplar_idx]
        instance_img_path = valid_img_files[instance_idx]
        instance_anno = valid_annos[instance_idx]
        
        size_ratio = self.cfg.exemplar_size / self.cfg.instance_size
        exemplar_img = self._read_image_and_transform(
            exemplar_img_path, exemplar_anno, self.cfg.exemplar_size,
            self.transform_exemplar, size_ratio)
        instance_img = self._read_image_and_transform(
            instance_img_path, instance_anno, self.cfg.instance_size,
            self.transform_instance)
        
        return exemplar_img, instance_img
    
    def __len__(self) -> int:
        return len(self.data_seq) * self.cfg.pairs_per_seq
    
    def _sample_pair_indices(self, n_items: int) -> Tuple[int, int]:
        max_distance = min(n_items - 1, self.cfg.max_pair_dist)
        rand_indices = np.random.choice(max_distance + 1, 2)
        rand_start = np.random.randint(n_items - max_distance)
        rand_indices = rand_indices + rand_start
        
        if rand_indices[1] < rand_indices[0]:
            rand_indices[0], rand_indices[1] = rand_indices[1], rand_indices[0]
        
        return rand_indices
    
    def _filter_valid_indices(self, annos: np.ndarray) -> np.ndarray:
        side_lengths = annos[:, 2:]
        valid_indices = side_lengths.prod(axis=1) >= self.cfg.min_bbox_area
        return valid_indices
    
    @staticmethod
    def _read_image_and_transform(
            img_path: str, anno: np.ndarray, output_side_size: int,
            transform: Callable[[Image.Image], torch.Tensor],
            size_with_context_scale: float = 1.0) -> torch.Tensor:
        bbox = BBox(*anno)
        side_size_with_context = calc_bbox_side_size_with_context(bbox)
        side_size_scaled = side_size_with_context * size_with_context_scale
        new_size = (side_size_scaled, side_size_scaled)
        bbox.size = np.asarray(new_size).round().astype(np.int)  # !!!
        
        img = Image.open(img_path)
        output_side = (output_side_size, output_side_size)
        patch = center_crop_and_resize(img, bbox, output_side)
        
        if patch.mode == 'L':
            patch = patch.convert('RGB')
        patch_tensor = transform(patch)
        
        return patch_tensor
    
    @staticmethod
    def _build_transforms(
            output_size: int, *, max_translate: int = 4,
            max_stretch: float = 0.05):
        return T.Compose([
            T.RandomHorizontalFlip(0.2),
            RandomStretch(max_stretch),
            T.RandomCrop(
                output_size, padding=max_translate, pad_if_needed=True,
                padding_mode='edge'),
            T.ToTensor()])


class SiamesePairwiseWithTimeDataset(Dataset):
    def __init__(self, data_seq: Sequence, cfg: TrackerConfig,
                 time_point: int = 25, time_weight: float = 0.5) -> None:
        super().__init__()
        
        self.data_seq: Sequence = data_seq
        self.cfg: TrackerConfig = cfg
        
        self.indices: np.ndarray = np.random.permutation(len(self.data_seq))
        
        self.transform_exemplar = self._build_transforms(self.cfg.exemplar_size)
        self.transform_instance = self._build_transforms(self.cfg.instance_size)
        
        self.time_weight_decay = (1 / time_point) * np.log(1 - time_weight)
    
    def __getitem__(self, index: int) -> PairItemT:
        index = self.indices[index % len(self.data_seq)]
        img_files, annos = self.data_seq[index]
        annos = annos.astype(np.int)
        
        valid_indices = self._filter_valid_indices(annos)
        valid_img_files = np.asarray(img_files)[valid_indices]
        valid_annos = annos[valid_indices, :]
        
        n_imgs = len(valid_img_files)
        exemplar_idx, instance_idx = self._sample_pair_indices(n_imgs)
        
        exemplar_img_path = valid_img_files[exemplar_idx]
        exemplar_anno = valid_annos[exemplar_idx]
        instance_img_path = valid_img_files[instance_idx]
        instance_anno = valid_annos[instance_idx]
        
        size_ratio = self.cfg.exemplar_size / self.cfg.instance_size
        exemplar_img = self._read_image_and_transform(
            exemplar_img_path, exemplar_anno, self.cfg.exemplar_size,
            self.transform_exemplar, size_ratio)
        instance_img = self._read_image_and_transform(
            instance_img_path, instance_anno, self.cfg.instance_size,
            self.transform_instance)
        
        exemplar_img = self._add_time_dimension(exemplar_img, 0)
        time_diff = instance_idx - exemplar_idx
        instance_img = self._add_time_dimension(instance_img, time_diff)
        
        return exemplar_img, instance_img
    
    def __len__(self) -> int:
        return len(self.data_seq) * self.cfg.pairs_per_seq
    
    def _sample_pair_indices(self, n_items: int) -> Tuple[int, int]:
        max_distance = min(n_items - 1, self.cfg.max_pair_dist)
        rand_indices = np.random.choice(max_distance + 1, 2)
        rand_start = np.random.randint(n_items - max_distance)
        rand_indices = rand_indices + rand_start
        
        if rand_indices[1] < rand_indices[0]:
            rand_indices[0], rand_indices[1] = rand_indices[1], rand_indices[0]
        
        return rand_indices
    
    def _filter_valid_indices(self, annos: np.ndarray) -> np.ndarray:
        side_lengths = annos[:, 2:]
        valid_indices = side_lengths.prod(axis=1) >= self.cfg.min_bbox_area
        return valid_indices
    
    @staticmethod
    def _read_image_and_transform(
            img_path: str, anno: np.ndarray, output_side_size: int,
            transform: Callable[[Image.Image], torch.Tensor],
            size_with_context_scale: float = 1.0) -> torch.Tensor:
        bbox = BBox(*anno)
        side_size_with_context = calc_bbox_side_size_with_context(bbox)
        side_size_scaled = side_size_with_context * size_with_context_scale
        new_size = (side_size_scaled, side_size_scaled)
        bbox.size = np.asarray(new_size).round().astype(np.int)  # !!!
        
        img = Image.open(img_path)
        output_side = (output_side_size, output_side_size)
        patch = center_crop_and_resize(img, bbox, output_side)
        
        if patch.mode == 'L':
            patch = patch.convert('RGB')
        patch_tensor = transform(patch)
        
        return patch_tensor
        
    def _add_time_dimension(
            self, img: torch.Tensor, time: float) -> torch.Tensor:
        weight = 1 - np.exp(self.time_weight_decay * time)
        
        weight_dim = torch.full((1, *img.shape[1:]), weight)
        time_weighted_img = torch.vstack((img, weight_dim))
        
        return time_weighted_img
    
    @staticmethod
    def _build_transforms(
            output_size: int, *, max_translate: int = 4,
            max_stretch: float = 0.05):
        # return T.Compose([
        #     T.RandomHorizontalFlip(0.2),
        #     RandomStretch(max_stretch),
        #     T.RandomCrop(
        #         output_size, padding=max_translate, pad_if_needed=True,
        #         padding_mode='edge'),
        #     T.ToTensor()])
        return T.ToTensor()


def build_dataset_and_init(cls, *args, **kwargs):
    inst = cls(*args, **kwargs)
    inst.initialize()
    return inst


if __name__ == '__main__':
    from typing import cast, Sequence
    from got10k.datasets import GOT10k
    
    from sot.utils import cv_show_tensor_as_img
    dataset = GOT10k(root_dir="../../../../datasets/GOT10k", subset='val')
    pairwise_dataset = SiamesePairwiseDataset(
        cast(Sequence, dataset), TrackerConfig())
    count = 10
    
    for i in range(count):
        exemplar_img, instance_img = pairwise_dataset[i]
        cv_show_tensor_as_img(exemplar_img, "exemplar")
        cv_show_tensor_as_img(instance_img, "instance")
        cv.waitKey(0)
    
    cv.destroyAllWindows()

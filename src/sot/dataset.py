import os
import abc
import pickle
import pathlib
import dataclasses
import collections
import xml.etree.ElementTree as ET

import cv2 as cv
import numpy as np
import torch

from typing import (
    Sequence, Iterable, Tuple, Union, Optional, DefaultDict, List, Callable)

from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor

from sot.bbox import BBox
from sot.cfg import TrackerConfig
from sot.utils import center_crop_and_resize, calc_bbox_side_size_with_context


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
        self.track_data: DefaultDict[str, List[TrackData]] =\
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
                
                for seq_dir in anno_subset_dir.rglob(f'ILSVRC2015_{subset}_*'):
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
                for track_id, img_file_path in\
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
        img_file_path = os.path.join(folder, f'{filename}.JPEG')
        
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


class OTBDataset:
    pass


PairItemT = Tuple[torch.Tensor, torch.Tensor]


class SiamesePairwiseDataset(Dataset):
    def __init__(self, data_seq: Sequence, cfg: TrackerConfig) -> None:
        super().__init__()

        self.data_seq: Sequence = data_seq
        self.cfg: TrackerConfig = cfg
        
        self.indices: np.ndarray = np.random.permutation(len(self.data_seq))
        
        # TODO Add data augmentation.
        self.transform_exemplar = Compose([ToTensor()])
        self.transform_instance = Compose([ToTensor()])
    
    def __getitem__(self, index: int) -> PairItemT:
        assert index >= 0
        
        index = self.indices[index % len(self.data_seq)]
        img_files, annos = self.data_seq[index]
        
        valid_indices = annos[:, 2:].prod(axis=1) >= self.cfg.min_bbox_area
        valid_img_files = np.asarray(img_files)[valid_indices]
        valid_annos = annos[valid_indices, :]
        assert len(valid_img_files) == len(valid_annos)
        
        n_imgs = len(valid_img_files)
        exemplar_idx, instance_idx = self.sample_pair_indices(n_imgs)
        
        exemplar_img_path = valid_img_files[exemplar_idx]
        exemplar_anno = valid_annos[exemplar_idx]
        instance_img_path = valid_img_files[instance_idx]
        instance_anno = valid_annos[instance_idx]

        size_ratio = self.cfg.exemplar_size / self.cfg.instance_size
        assert self.cfg.exemplar_size < self.cfg.instance_size
        exemplar_img = self.read_image_and_transform(
            exemplar_img_path, exemplar_anno, self.cfg.exemplar_size,
            self.transform_exemplar, size_ratio)
        instance_img = self.read_image_and_transform(
            instance_img_path, instance_anno, self.cfg.instance_size,
            self.transform_instance)
        
        return exemplar_img, instance_img
    
    def __len__(self) -> int:
        return len(self.data_seq) * self.cfg.pairs_per_seq
    
    def sample_pair_indices(self, count: int) -> Tuple[int, int]:
        assert count > 0
        
        max_distance = min(count - 1, self.cfg.max_pair_dist)
        rand_indices = np.random.choice(max_distance + 1, 2)
        rand_start = np.random.randint(count - max_distance)
        return rand_indices + rand_start
    
    @staticmethod
    def read_image_and_transform(
            img_path: str, anno: np.ndarray, output_side_size: int,
            transform: Callable[[np.ndarray], torch.Tensor],
            size_with_context_scale: float = 1.0) -> torch.Tensor:
        assert output_side_size > 0
        assert size_with_context_scale > 0
        
        bbox = BBox(*anno)
        side_size_with_context = calc_bbox_side_size_with_context(bbox)
        side_size_scaled = side_size_with_context * size_with_context_scale
        new_size = (side_size_scaled, side_size_scaled)
        bbox.size = np.asarray(new_size).round().astype(np.int)  # !!!
        
        img = cv.imread(img_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        
        output_side = (output_side_size, output_side_size)
        patch = center_crop_and_resize(img, bbox, output_side)
        
        patch_tensor = transform(patch)
        
        return patch_tensor


def build_dataset_and_init(cls, *args, **kwargs):
    inst = cls(*args, **kwargs)
    inst.initialize()
    return inst


if __name__ == '__main__':
    cache_file = pathlib.Path('../../dataset_train_dump.bin')
    if cache_file.exists():
        with open(str(cache_file), 'rb') as in_file:
            data_seq = pickle.load(in_file)
    else:
        dataset_path = '../../../../datasets/ILSVRC2015_VID_small'
        data_seq = build_dataset_and_init(
            ImageNetVideoDataset, dataset_path, 'train')
        with open(str(cache_file), 'wb') as out_file:
            pickle.dump(data_seq, out_file, protocol=pickle.HIGHEST_PROTOCOL)
    
    pairwise_dataset = SiamesePairwiseDataset(data_seq, TrackerConfig())
    count = 10
    
    for i in range(count):
        exemplar_img, instance_img = pairwise_dataset[i]
        
        exemplar_img = (exemplar_img.numpy() * 255).astype(np.uint8)
        exemplar_img = np.transpose(exemplar_img, axes=(1, 2, 0))
        
        instance_img = (instance_img.numpy() * 255).astype(np.uint8)
        instance_img = np.transpose(instance_img, axes=(1, 2, 0))
        
        cv.imshow('exemplar', exemplar_img)
        cv.imshow('instance', instance_img)
        cv.waitKey(0)
    
    cv.destroyAllWindows()
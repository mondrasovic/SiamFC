import os
import abc
import pickle
import pathlib
import collections
import xml.etree.ElementTree as ET

from typing import Sequence, Iterable, Tuple, Union, DefaultDict, List, Optional

import numpy as np

BBoxT = Tuple[int, int, int, int]
TrackDataT = DefaultDict[str, List[Tuple[str, Optional[BBoxT]]]]


class Dataset(collections.abc.Sequence):
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


class ImageNetVideoDataset(Dataset):
    def __init__(
            self, root_dir_path: str,
            subsets: Union[str, Sequence[str]] = 'train'):
        super().__init__('ImageNetVID', root_dir_path)
        
        if isinstance(subsets, str):
            subsets = (subsets,)
        assert all(subset in ('train', 'val', 'test') for subset in subsets)
        self.subsets: Sequence[str] = subsets
        
        self.track_data: TrackDataT = collections.defaultdict(list)
    
    def read_track_ids(self) -> Iterable[str]:
        # In this dataset, one video may contain multiple track. Even though
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
                            self.track_data[track_id_ex].append(
                                (str(img_file_path_ex), bbox))
            else:  # 'test'
                test_data_dir = root_dir / 'Data' / 'VID' / 'test'
                for track_id, img_file_path in\
                        self._read_test_track_ids_and_img_paths(test_data_dir):
                    self.track_data[track_id].append((img_file_path, None))
        return self.track_data.keys()

    def read_img_file_paths(self, track_id: str) -> Iterable[str]:
        return (data[0] for data in self.track_data[track_id])

    def read_annotations(self, track_id) -> np.ndarray:
        annos = [data[1] for data in self.track_data[track_id]]
        return np.array(annos, dtype=np.int)
    
    @staticmethod
    def _read_annos_from_xml(
            xml_file_path: str) -> Iterable[Tuple[str, str, BBoxT]]:
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
            bbox = (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)
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


def build_dataset(cls, *args, **kwargs):
    inst = cls(*args, **kwargs)
    inst.initialize()
    return inst


if __name__ == '__main__':
    cache_file = pathlib.Path('../../dataset_train_dump.bin')
    if cache_file.exists():
        with open(str(cache_file), 'rb') as in_file:
            dataset = pickle.load(in_file)
    else:
        dataset_path = '../../../../datasets/ILSVRC2015_VID_small'
        dataset = build_dataset(ImageNetVideoDataset, dataset_path, 'train')
        with open(str(cache_file), 'wb') as out_file:
            pickle.dump(dataset, out_file, protocol=pickle.HIGHEST_PROTOCOL)

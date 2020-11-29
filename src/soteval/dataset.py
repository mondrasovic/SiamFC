import abc
import collections
import dataclasses

from typing import Sequence, Iterator

import numpy as np


@dataclasses.dataclass(frozen=True)
class TrackSequence:
    img_file_paths: Sequence[str]
    annos: np.ndarray


class Dataset(collections.abc.Sequence):
    def __init__(self, name: str, root_dir_path: str) -> None:
        self.name: str = name
        self.root_dir_path: str = root_dir_path
        
        self.seq_names: Sequence[str] = tuple(self._read_seq_names())
    
    def __getitem__(self, index: int) -> TrackSequence:
        seq_name = self.seq_names[index]
        img_file_paths = tuple(self._read_img_file_paths(seq_name))
        annos = self._read_annotations(seq_name)
        track_seq = TrackSequence(img_file_paths, annos)
        return track_seq
        
    def __len__(self) -> int:
        return len(self.seq_names)
    
    @abc.abstractmethod
    def _read_seq_names(self) -> Iterator[str]:
        pass
    
    @abc.abstractmethod
    def _read_img_file_paths(self, seq_name: str) -> Iterator[str]:
        pass
    
    @abc.abstractmethod
    def _read_annotations(self, seq_name) -> np.ndarray:
        pass


class ImageNetVideoDataset:
    pass


class OTB2013Dataset:
    pass


class OTB2015Dataset:
    pass

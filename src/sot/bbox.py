import numpy as np

from typing import Optional


class BBox:
    def __init__(self, x: int, y: int, width: int, height: int) -> None:
        assert min(width, height) >= 0, "width and height must be non-negative"
        
        self._center: np.ndarray = np.asarray((x + width // 2, y + height // 2))
        self._size: np.ndarray = np.asarray((width, height))
    
    @property
    def size(self) -> np.ndarray:
        return self._size
    
    @property
    def center(self) -> np.ndarray:
        return self._center
    
    @size.setter
    def size(self, new_size: np.ndarray) -> None:
        assert (new_size.ndim == 1) and (len(new_size) == 2)
        assert new_size.min() >= 0, "width and height must be non-negative"
        
        self._size = new_size
    
    @staticmethod
    def build_from_center_and_size(
            center: np.ndarray, size: np.ndarray) -> 'BBox':
        x, y = center - size // 2
        return BBox(x, y, *size)
    
    def as_corners(self) -> np.ndarray:
        xy = self.center - self.size // 2
        return np.concatenate((xy, xy + self.size))
    
    def as_xywh(self) -> np.ndarray:
        xy = self.center - self.size // 2
        return np.concatenate((xy, self.size))
    
    def shift(
            self, center_shift: np.ndarray, in_place=True) -> Optional['BBox']:
        assert (center_shift.ndim == 1) and (len(center_shift) == 2)
        
        new_center = self._center + center_shift
        if in_place:
            self._center = new_center
            return None
        else:
            return BBox.build_from_center_and_size(new_center, self.size)
    
    def rescale(
            self, width_scale: float, height_scale: float,
            in_place=True) -> Optional['BBox']:
        assert min(width_scale, height_scale) >= 0,\
            "width and height scale factors must be non-negative"
        
        new_size = self.size * np.array((width_scale, height_scale))
        new_size = new_size.round().astype(np.int)
        
        if in_place:
            self.size = new_size
            return None
        else:
            return BBox.build_from_center_and_size(self.center, new_size)
    
    def __repr__(self) -> str:
        x, y = self.center - self.size // 2
        width, height = self.size
        return f'BBox({x},{y},{width},{height})'
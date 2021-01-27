import numpy as np

from typing import Optional


class BBox:
    def __init__(self, x: int, y: int, width: int, height: int) -> None:
        assert min(width, height) >= 0, "width and height must be non-negative"
        
        self.center: np.ndarray = np.asarray((x + width // 2, y + height // 2))
        self.size: np.ndarray = np.asarray((width, height))
    
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

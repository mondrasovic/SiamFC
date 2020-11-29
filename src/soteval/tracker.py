import abc

import numpy as np

from PIL import Image

BBoxT = np.ndarray


class Tracker:
    def __init__(self, name: str) -> None:
        self.name: str = name
    
    @abc.abstractmethod
    def init(self, img: Image, bbox: BBoxT) -> None:
        pass
    
    @abc.abstractmethod
    def update(self, img: Image) -> BBoxT:
        pass

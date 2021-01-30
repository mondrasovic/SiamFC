import abc
import click
import numbers

import cv2 as cv
import numpy as np

from sot.bbox import BBox


IMG_WIDTH = 1024
IMG_HEIGHT = 768

N_TRACKED_OBJECTS = 2


def rand_between(a, b, *, size=None, as_int=False):
    diff = b - a
    
    if size is None:
        vals = np.random.rand() * diff + a
    elif isinstance(size, numbers.Number):
        vals = np.random.rand(size) * diff + a
    else:
        vals = np.random.rand(*size) * diff + a
    
    return np.round(vals).astype(np.int) if as_int else vals


class MotionModel:
    def __init__(
            self, bbox, img_size, *, friction=0.5, center_attraction=0.05):
        self.bbox = bbox
        self.img_size = img_size
        self.friction = friction
        self.center_attraction = center_attraction
        
        self.velocity = np.zeros(2, dtype=np.float)
        self.min_bbox_side = self.bbox.size.min() // 2
        self.max_bbox_side = int(round(self.bbox.size.max() * 1.5))
        self.img_center = self.img_size / 2
    
    def update(self, shift, repulsion_points=None):
        self.velocity = self.friction * self.velocity +\
                        (1 - self.friction) * shift
        
        center_diff = (self.img_center - self.bbox.center)
        self.velocity = (1 - self.center_attraction) * self.velocity +\
                        self.center_attraction * center_diff
        
        if repulsion_points:
            pass
        
        center_shift = self.velocity.round().astype(np.int)
        bbox_shifted = self.bbox.shift(center_shift, in_place=False)
        corners = bbox_shifted.as_corners()
        if (corners[:2] >= 0).all() and (corners[3:] < self.img_size).all():
            self.bbox = bbox_shifted
    
    def rescale(self, width_scale, height_scale):
        bbox_scaled = self.bbox.rescale(
            width_scale, height_scale, in_place=False)
        size = bbox_scaled.size
        if ((size >= self.min_bbox_side) & (size <= self.max_bbox_side)).all():
            self.bbox = bbox_scaled


class TrackedObject(abc.ABC):
    def __init__(self, motion_model, color):
        self.motion_model = motion_model
        self.draw_kwargs = dict(color=color, thickness=-1, lineType=cv.LINE_AA)
    
    def move(self, center_shift):
        self.motion_model.update(center_shift)
    
    def rescale(self, width_scale, height_scale):
        self.motion_model.rescale(width_scale, height_scale)
    
    @abc.abstractmethod
    def render(self, img):
        pass


class Ellipse(TrackedObject):
    def __init__(self, motion_model, color):
        super().__init__(motion_model, color)
    
    def render(self, img):
        center = tuple(self.motion_model.bbox.center)
        half_size = tuple(self.motion_model.bbox.size // 2)
        
        cv.ellipse(
            img, center, half_size, startAngle=0, endAngle=0,
            **self.draw_kwargs)


class Rectangle(TrackedObject):
    
    def __init__(self, motion_model, color):
        super().__init__(motion_model, color)
    
    def render(self, img):
        tl, br = self.motion_model.bbox.as_tl_br()
        tl, br = tuple(tl), tuple(br)
        
        cv.rectangle(img, tl, br, **self.draw_kwargs)


class TrackedObjectsManager:
    def __init__(
            self, tracked_objs, coord_move_range=(-50, 50),
            side_scale_range=(0.95, 1.05)):
        self.tracked_objs = tracked_objs
        
        self.coord_move_range = coord_move_range
        self.side_scale_range = side_scale_range
    
    def move(self):
        for tracked_obj in self.tracked_objs:
            center_shift = rand_between(*self.coord_move_range, size=2)
            tracked_obj.move(center_shift)
    
    def rescale(self):
        for tracked_obj in self.tracked_objs:
            width_scale, height_scale = rand_between(
                *self.side_scale_range, size=2)
            tracked_obj.rescale(width_scale, height_scale)
    
    def render(self, img):
        for tracked_obj in self.tracked_objs:
            tracked_obj.render(img)


class TrackedObjectGenerator:
    SUPPORTED_OBJECTS = {
        'rectangle': Rectangle,
        'ellipse': Ellipse,
    }
    
    def __init__(
            self, img_size, width_range=(80, 100), height_range=(40, 60),
            friction_range=(0.4, 0.8), center_attraction_range=(0.03, 0.07)):
        self.img_size = img_size
        
        self.width_range = width_range
        self.height_range = height_range
        
        self.friction_range = friction_range
        self.center_attraction_range = center_attraction_range
    
    def generate_tracked_object(self, obj_name):
        obj_cls = self.SUPPORTED_OBJECTS[obj_name]
        motion_model = self.generate_motion_model()
        color = tuple(map(int, rand_between(0, 255, size=3, as_int=True)))
        
        return obj_cls(motion_model, color)
    
    def generate_motion_model(self):
        friction = rand_between(*self.friction_range)
        center_attraction = rand_between(*self.center_attraction_range)
        bbox = self.generate_bbox()
        
        return MotionModel(
            bbox, self.img_size, friction=friction,
            center_attraction=center_attraction)
    
    def generate_bbox(self):
        valid_x_range = 0, self.img_size[0] - self.width_range[1] - 1
        valid_y_range = 0, self.img_size[1] - self.height_range[1] - 1
        
        x = rand_between(*valid_x_range, as_int=True)
        y = rand_between(*valid_y_range, as_int=True)
        
        width = rand_between(*self.width_range, as_int=True)
        height = rand_between(*self.height_range, as_int=True)
        
        return BBox(x, y, width, height)


@click.command()
@click.argument('output_dir_path')
def main(output_dir_path):
    np.random.seed(731995)
    
    img_size = np.asarray((IMG_WIDTH, IMG_HEIGHT))
    obj_gen = TrackedObjectGenerator(img_size)
    tracked_objs = [
        obj_gen.generate_tracked_object('rectangle')
        for _ in range(N_TRACKED_OBJECTS)]
    tracked_objs_man = TrackedObjectsManager(tracked_objs)
    
    for _ in range(500):
        img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        
        tracked_objs_man.move()
        tracked_objs_man.rescale()
        tracked_objs_man.render(img)
        
        cv.imshow('preview', img)
        key = cv.waitKey(60) & 0xff
        if key == ord('q'):
            break
    
    cv.destroyAllWindows()
    
    return 0


if __name__ == '__main__':
    import sys
    
    sys.exit(main())

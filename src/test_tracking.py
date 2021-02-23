#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Milan Ondrasovic <milan.ondrasovic@gmail.com>

import sys
import pathlib
from typing import Iterable, Optional, cast

import click
import cv2 as cv
import numpy as np
import torch

from sot.cfg import TrackerConfig
from sot.tracker import TrackerSiamFC
from sot.utils import cv_to_pil_img, ImageT, pil_to_cv_img
from sot.visual import SiameseTrackingVisualizer


def iter_video_capture() -> Iterable[np.ndarray]:
    cap = cv.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame


def is_image_file(file: pathlib.Path) -> bool:
    return file.suffix.lower() in (".jpg", ".jpeg", ".png")


def iter_dir_imgs(dir_path: str) -> Iterable[np.ndarray]:
    for file in filter(is_image_file, pathlib.Path(dir_path).iterdir()):
        img = cv.imread(str(file), cv.IMREAD_COLOR)
        yield img


@click.command()
@click.option("-i", "--imgs-dir-path", help="directory path with images")
@click.option("-m", "--model-file-path", help="a pre-trained model file path")
def main(imgs_dir_path: Optional[str], model_file_path: Optional[str]) -> int:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = TrackerConfig()
    tracker = TrackerSiamFC(cfg, device, model_file_path)
    
    curr_exemplar_img = None
    curr_instance_img = None
    curr_response_map = None
    
    def retrieve_exemplar_img(exemplar_img: ImageT) -> None:
        nonlocal curr_exemplar_img
        curr_exemplar_img = pil_to_cv_img(exemplar_img)
    
    def retrieve_instance_img(instance_img: ImageT) -> None:
        nonlocal curr_instance_img
        curr_instance_img = pil_to_cv_img(instance_img)
    
    def retrieve_response_map(response_map: np.ndarray) -> None:
        nonlocal curr_response_map
        curr_response_map = response_map
    
    tracker.on_exemplar_img_extract = retrieve_exemplar_img
    tracker.on_instance_img_extract = retrieve_instance_img
    tracker.on_response_map_calc = retrieve_response_map

    if imgs_dir_path is None:
        imgs_iter = iter_video_capture()
    else:
        imgs_iter = iter_dir_imgs(imgs_dir_path)
    is_first = True
    
    visualizer = None
    
    for frame in imgs_iter:
        if is_first:
            # bbox = np.asarray(cv.selectROI("tracker initialization", frame))
            bbox = np.asarray((765, 389, 517, 470))
            tracker.init(cv_to_pil_img(frame), bbox)
            visualizer = SiameseTrackingVisualizer(
                cast(np.ndarray, curr_exemplar_img), border_value=(32, 32, 32))
            is_first = False
        else:
            bbox_pred = tracker.update(cv_to_pil_img(frame))
            if not visualizer.show_curr_state(
                frame, cast(np.ndarray, curr_instance_img),
                cast(np.ndarray, curr_response_map), bbox_pred):
                break
    
    visualizer.close()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

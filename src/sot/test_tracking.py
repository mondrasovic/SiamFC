import pathlib
from typing import Iterable, Optional

import click
import cv2 as cv
import numpy as np
import torch

from sot.cfg import TrackerConfig, MODEL_DIR
from sot.tracker import TrackerSiamFC


def iter_video_capture() -> Iterable[np.ndarray]:
    cap = cv.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        yield frame


def iter_dir_imgs(dir_path: str) -> Iterable[np.ndarray]:
    for file in pathlib.Path(dir_path).iterdir():
        img = cv.imread(str(file), cv.IMREAD_COLOR)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        yield img


@click.command()
@click.option("-i", "--imgs-dir-path", help="directory path with images")
def main(imgs_dir_path: Optional[str]) -> int:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = TrackerConfig()
    tracker = TrackerSiamFC(cfg, device, MODEL_DIR)
    
    is_first = True
    
    if imgs_dir_path is None:
        imgs_iter = iter_video_capture()
    else:
        imgs_iter = iter_dir_imgs(imgs_dir_path)
    
    for frame in imgs_iter:
        if is_first:
            bbox = cv.selectROI("tracker initialization", frame)
            # bbox = (463,288,131,101)
            bbox = np.asarray(bbox)
            tracker.init(frame, bbox)
            is_first = False
        else:
            bbox_pred = tracker.update(frame)
            pt1 = tuple(bbox_pred[:2])
            pt2 = tuple(bbox_pred[:2] + bbox_pred[2:])
            cv.rectangle(frame, pt1, pt2, (0, 255, 0), 3, cv.LINE_AA)
        
        cv.imshow("tracking preview", frame)
        key = cv.waitKey(1) & 0xff
        if key == ord('q'):
            break
    
    return 0


if __name__ == '__main__':
    import sys
    
    sys.exit(main())

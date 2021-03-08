#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Milan Ondrasovic <milan.ondrasovic@gmail.com>

import sys
import pathlib
from typing import Optional

import click
import torch

from common import DatasetType, init_experiment
from sot.cfg import TrackerConfig
from sot.tracker import TrackerSiamFC


@click.command()
@click.argument("dataset_name")
@click.argument("dataset_dir_path", type=click.Path(exists=True))
@click.argument("results_dir_path", type=click.Path())
@click.argument("reports_dir_path", type=click.Path())
@click.option(
    "-d", "--models-dir-path", type=click.Path(exists=True),
    help="directory path with pre-trained models")
@click.option(
    "--tracker-name",
    help="tracker name (for producing results and reports) to which a suffix "
         "will be added")
def main(
        dataset_name: str, dataset_dir_path: str, results_dir_path: str,
        reports_dir_path: str,
        models_dir_path: Optional[str], tracker_name: Optional[str]) -> int:
    """
    Starts a SiamFC evaluation with the specific DATASET_NAME
    (GOT10k | OTB13 | OTB15 | VOT15 | UAV123) located in the DATASET_DIR_PATH.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = TrackerConfig()
    
    dataset_type = DatasetType.decode_dataset_type(dataset_name)
    experiment = init_experiment(
        dataset_type, dataset_dir_path, results_dir_path, reports_dir_path)
    
    for model_file in pathlib.Path(models_dir_path).iterdir():
        suffix = model_file.stem
        curr_tracker_name = "" if tracker_name is None else tracker_name
        curr_tracker_name += suffix
        
        tracker = TrackerSiamFC(
            cfg, device, str(model_file), name=curr_tracker_name)
        
        experiment.run(tracker, visualize=False)
        experiment.report([tracker.name])
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

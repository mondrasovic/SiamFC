#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Milan Ondrasovic <milan.ondrasovic@gmail.com>

import sys
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
    "-m", "--model-file-path", type=click.Path(exists=True),
    help="a pre-trained model file path")
@click.option(
    "--tracker-name", help="tracker name (for producing results and reports)")
def main(
        dataset_name: str, dataset_dir_path: str, results_dir_path: str,
        reports_dir_path: str,
        model_file_path: Optional[str], tracker_name: Optional[str]) -> int:
    """
    Starts a SiamFC evaluation with the specific DATASET_NAME
    (GOT10k | OTB13 | OTB15 | VOT15 | UAV123) located in the DATASET_DIR_PATH.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = TrackerConfig()
    
    dataset_type = DatasetType.decode_dataset_type(dataset_name)
    tracker = TrackerSiamFC(cfg, device, model_file_path, name=tracker_name)
    experiment = init_experiment(
        dataset_type, dataset_dir_path, results_dir_path, reports_dir_path)
    
    experiment.run(tracker, visualize=False)
    experiment.report([tracker.name])
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

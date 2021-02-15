#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Milan Ondrasovic <milan.ondrasovic@gmail.com>

import sys
from typing import Optional

import click
import torch
from got10k.experiments import ExperimentOTB, ExperimentGOT10k

from common import DatasetType
from sot.cfg import TrackerConfig
from sot.tracker import TrackerSiamFC


def init_experiment(
        dataset_type: DatasetType, dataset_dir_path: str, results_dir_path: str,
        reports_dir_path: str):
    params = dict(
        root_dir=dataset_dir_path, result_dir=results_dir_path,
        reports_dir=reports_dir_path)
    
    if dataset_type == DatasetType.OTB13:
        return ExperimentOTB(version=2013, **params)
    elif dataset_type == DatasetType.OTB15:
        return ExperimentOTB(version=2015, **params)
    elif dataset_type == DatasetType.GOT10K:
        return ExperimentGOT10k(subset='val', **params)
    else:
        raise ValueError(f"unsupported dataset type {dataset_type}")


@click.command()
@click.argument("dataset_name")
@click.argument("dataset_dir_path")
@click.argument("results_dir_path")
@click.argument("reports_dir_path")
@click.option("-m", "--model-file-path", help="a pre-trained model file path")
def main(
        dataset_name: str, dataset_dir_path: str, results_dir_path: str,
        reports_dir_path: str,
        model_file_path: Optional[str]) -> int:
    """
    Starts a SiamFC evaluation with the specific DATASET_NAME
    (GOT10k | OTB13 | OTB15) located in the DATASET_DIR_PATH.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = TrackerConfig()
    
    dataset_type = DatasetType.decode_dataset_type(dataset_name)
    tracker = TrackerSiamFC(cfg, device, model_file_path)
    experiment = init_experiment(
        dataset_type, dataset_dir_path, results_dir_path, reports_dir_path)
    
    experiment.run(tracker, visualize=False)
    experiment.report([tracker.name])
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Milan Ondrasovic <milan.ondrasovic@gmail.com>

import enum

from got10k.experiments import (
    ExperimentOTB, ExperimentGOT10k, ExperimentVOT, ExperimentUAV123)


@enum.unique
class DatasetType(enum.Enum):
    GOT10k = 'GOT10k'
    OTB13 = 'OTB13'
    OTB15 = 'OTB15'
    VOT15 = 'VOT15'
    UAV123 = 'UAV123'
    ILSVRC15 = 'ILSVRC15'
    
    @staticmethod
    def decode_dataset_type(dataset_name: str) -> 'DatasetType':
        for dataset_item in DatasetType:
            if dataset_item.value == dataset_name:
                return dataset_item
        raise ValueError("unrecognized dataset type")


def init_experiment(
        dataset_type: DatasetType, dataset_dir_path: str, results_dir_path: str,
        reports_dir_path: str):
    params = dict(
        root_dir=dataset_dir_path, result_dir=results_dir_path,
        report_dir=reports_dir_path)
    
    if dataset_type == DatasetType.OTB13:
        return ExperimentOTB(version=2013, **params)
    elif dataset_type == DatasetType.OTB15:
        return ExperimentOTB(version=2015, **params)
    elif dataset_type == DatasetType.GOT10k:
        return ExperimentGOT10k(subset='val', **params)
    elif dataset_type == DatasetType.VOT15:
        return ExperimentVOT(version=2015, **params)
    elif dataset_type == DatasetType.UAV123:
        return ExperimentUAV123(**params)
    else:
        raise ValueError(f"unsupported dataset type {dataset_type}")

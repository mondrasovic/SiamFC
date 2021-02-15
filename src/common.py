#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Milan Ondrasovic <milan.ondrasovic@gmail.com>

import enum


@enum.unique
class DatasetType(enum.Enum):
    GOT10K = 'GOT10k'
    OTB13 = 'OTB13'
    OTB15 = 'OTB15'
    VOT15 = 'VOT15'
    
    @staticmethod
    def decode_dataset_type(dataset_name: str) -> 'DatasetType':
        for dataset_item in DatasetType:
            if dataset_item.value == dataset_name:
                return dataset_item
        raise ValueError("unrecognized dataset type")

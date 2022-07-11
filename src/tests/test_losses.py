#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Milan Ondrasovic <milan.ondrasovic@gmail.com>

import unittest

import torch

from sot.losses import WeightedBCELoss
from sot.utils import create_ground_truth_mask_and_weight


class TestLossFunction(unittest.TestCase):
    RESPONSE_MAP_SIZE = 272
    BATCH_SIZE = 8
    
    def setUp(self) -> None:
        size = (self.RESPONSE_MAP_SIZE, self.RESPONSE_MAP_SIZE)
        radius = self.RESPONSE_MAP_SIZE / 4
        total_stride = 8
        
        mask_mat, weight_mat = create_ground_truth_mask_and_weight(
            size, radius, total_stride, self.BATCH_SIZE)
        
        self.mask_mat = torch.from_numpy(mask_mat)
        weight_mat = torch.from_numpy(weight_mat)
        
        self.criterion = WeightedBCELoss(weight_mat)
        self.response_maps = torch.ones((self.BATCH_SIZE, 1, *size))
    
    def test_loss_output_shape(self):
        loss = self.criterion(self.response_maps, self.mask_mat)
        
        self.assertEqual(list(loss.shape), [])
        self.assertTrue(isinstance(loss.item(), float))


if __name__ == '__main__':
    unittest.main()

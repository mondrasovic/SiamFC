import unittest

import torch
import numpy as np

from sot.bbox import BBox
from sot.utils import (
    center_crop_and_resize, cv_img_to_tensor,
    create_ground_truth_mask_and_weight)


class TestCenterCropAndResize(unittest.TestCase):
    PIX_VALUE = 255
    
    def setUp(self) -> None:
        width, height = 800, 600
        self.img = np.full((height, width, 3), self.PIX_VALUE, dtype=np.uint8)
    
    def test_img_missing_channel_dim(self):
        with self.assertRaises(AssertionError):
            bbox = BBox(0, 0, 100, 100)
            center_crop_and_resize(np.ones((300, 300)), bbox, (100, 100))
    
    def test_output_shape(self):
        bbox = BBox(0, 0, 100, 100)
        width, height = 127, 255
        patch = center_crop_and_resize(self.img, bbox, (width, height))
        
        self.assertEqual(patch.shape, (height, width, 3))
    
    def test_bbox_within_image_no_padding(self):
        bbox = BBox(0, 0, 100, 100)
        patch = center_crop_and_resize(self.img, bbox, (127, 255))
        patch_flat = patch.flatten()
        
        self.assertTrue(np.all(patch_flat == patch_flat[0]))
    
    def test_bbox_outsize_image_needs_padding_constant_border(self):
        bbox = BBox(-50, -10, 200, 300)
        border = 127  # Value to fill the border with.
        patch = center_crop_and_resize(self.img, bbox, (127, 255), border)
        patch_flat = patch.flatten()
        values = np.unique(patch_flat)
        
        self.assertTrue(len(values) > 1)
        self.assertTrue(border in values)
        self.assertTrue(self.PIX_VALUE in values)


class TestCvImgToTensor(unittest.TestCase):
    PIX_VALUE = 255
    
    def setUp(self) -> None:
        width, height = 800, 600
        self.img = np.full((height, width, 3), self.PIX_VALUE, dtype=np.uint8)
        
    def test_wrong_n_dims_one_dim(self):
        with self.assertRaises(AssertionError):
            cv_img_to_tensor(np.ones(10))
    
    def test_wrong_n_dims_two_dims(self):
        with self.assertRaises(AssertionError):
            cv_img_to_tensor(np.ones((10, 20)))
    
    def test_wrong_n_dims_five_dims(self):
        with self.assertRaises(AssertionError):
            cv_img_to_tensor(np.ones((10, 20, 30, 40, 50)))
    
    def test_output_type(self):
        self.assertTrue(isinstance(cv_img_to_tensor(self.img), torch.Tensor))
    
    def test_output_n_dims(self):
        self.assertEqual(cv_img_to_tensor(self.img).ndim, 4)
    
    def test_output_dims_ordering(self):
        n_batches, width, height, n_channels = 10, 800, 600, 3
        tensor = cv_img_to_tensor(
            np.ones((n_batches, height, width, n_channels)))
        
        self.assertEqual(
            list(tensor.shape), [n_batches, n_channels, height, width])


class TestCreateGroundTruthMaskAndWeight(unittest.TestCase):
    RESPONSE_MAP_WIDTH = 255
    RESPONSE_MAP_HEIGHT = 127
    BATCH_SIZE = 8
    
    def setUp(self) -> None:
        size = (self.RESPONSE_MAP_WIDTH, self.RESPONSE_MAP_HEIGHT)
        radius = min(self.RESPONSE_MAP_WIDTH, self.RESPONSE_MAP_HEIGHT) / 4
        total_stride = 8
        
        self.mask_mat, self.weight_mat = create_ground_truth_mask_and_weight(
            size, radius, total_stride, self.BATCH_SIZE)
    
    def test_output_mask_mat_shape(self):
        expected_shape = (
            self.BATCH_SIZE, 1, self.RESPONSE_MAP_HEIGHT,
            self.RESPONSE_MAP_WIDTH)
        self.assertEqual(self.mask_mat.shape, expected_shape)
    
    def test_output_weight_mat_shape(self):
        self.assertEqual(
            self.weight_mat.shape,
            (1, self.RESPONSE_MAP_HEIGHT, self.RESPONSE_MAP_WIDTH))
    
    def test_mask_contains_only_ones_and_zeros(self):
        unique_values = np.unique(self.mask_mat)
        
        self.assertEqual(len(unique_values), 2)
        self.assertTrue(1 in unique_values)
        self.assertTrue(0 in unique_values)
    
    def test_mask_is_identical_for_each_batch(self):
        self.assertTrue((self.mask_mat == self.mask_mat[0]).all())
    
    def test_weight_matrix_contains_only_two_weights(self):
        unique_values = np.unique(self.weight_mat)
        
        self.assertEqual(len(unique_values), 2)
    
    def test_weight_matrix_sums_to_one(self):
        self.assertTrue(np.allclose(self.weight_mat.sum(), 1.0))


if __name__ == '__main__':
    unittest.main()

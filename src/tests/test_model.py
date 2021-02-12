import unittest

import numpy as np
import torch

from sot.model import SiamFCModel


class TestSiamFCModel(unittest.TestCase):
    def setUp(self) -> None:
        self.model = SiamFCModel()
    
    def test_feature_extraction_exemplar_shape(self):
        exemplar_img = torch.ones((1, 3, 127, 127))
        exemplar_emb = self.model.extract_visual_features(exemplar_img)
        
        self.assertEqual(list(exemplar_emb.shape), [1, 256, 6, 6])
    
    def test_feature_extraction_instance_shape(self):
        instance_img = torch.ones((1, 3, 255, 255))
        instance_emb = self.model.extract_visual_features(instance_img)
        
        self.assertEqual(list(instance_emb.shape), [1, 256, 22, 22])
    
    def test_forward_output_shape(self):
        exemplar_img = torch.ones((1, 3, 127, 127))
        instance_img = torch.ones((1, 3, 255, 255))
        response_map = self.model(exemplar_img, instance_img)
        
        self.assertEqual(list(response_map.shape), [1, 1, 17, 17])
    
    def test_response_map_shape_single_channel_single_instance(self):
        exemplar_emb = torch.ones(1, 1, 6, 6)
        instance_emb = torch.ones(1, 1, 22, 22)
        response_map = self.model.cross_corr(exemplar_emb, instance_emb)
        
        self.assertEqual(list(response_map.shape), [1, 1, 17, 17])
    
    def test_response_map_shape_single_channel_multiple_instances(self):
        exemplar_emb = torch.ones(10, 1, 6, 6)
        instance_emb = torch.ones(10, 1, 22, 22)
        response_map = self.model.cross_corr(exemplar_emb, instance_emb)
        
        self.assertEqual(list(response_map.shape), [10, 1, 17, 17])
    
    def test_response_map_shape_multiple_channels_single_instance(self):
        exemplar_emb = torch.ones(1, 10, 6, 6)
        instance_emb = torch.ones(1, 10, 22, 22)
        response_map = self.model.cross_corr(exemplar_emb, instance_emb)
        
        self.assertEqual(list(response_map.shape), [1, 1, 17, 17])
    
    def test_response_map_shape_multiple_channels_multiple_instances(self):
        exemplar_emb = torch.ones(10, 10, 6, 6)
        instance_emb = torch.ones(10, 10, 22, 22)
        response_map = self.model.cross_corr(exemplar_emb, instance_emb)
        
        self.assertEqual(list(response_map.shape), [10, 1, 17, 17])
    
    def test_cross_correlation_invalid_shapes(self):
        with self.assertRaises(AssertionError):
            exemplar_emb = torch.ones((1, 1, 1))
            instance_emb = torch.ones(1, 10, 5, 5)
            self.model.cross_corr(exemplar_emb, instance_emb)
    
    def test_cross_correlation_white_rectangle_search(self):
        def to_tensor(arr):
            return torch.from_numpy(
                np.transpose(arr, axes=(2, 0, 1))[None, ...])
        
        exemplar_size = 6
        instance_size = 22
        
        exemplar_emb = np.ones((exemplar_size, exemplar_size, 1))
        instance_emb = np.zeros((instance_size, instance_size, 1))
        
        exemplar_row, exemplar_col = 2, 10
        instance_emb[
        exemplar_row:exemplar_row + exemplar_size,
        exemplar_col:exemplar_col + exemplar_size, 0] = 1
        
        exemplar_emb = to_tensor(exemplar_emb)
        instance_emb = to_tensor(instance_emb)
        
        response = self.model.cross_corr(exemplar_emb, instance_emb).numpy()
        response = np.transpose(np.squeeze(response, axis=0), axes=(1, 2, 0))
        
        peak_response_idx = np.unravel_index(response.argmax(), response.shape)
        
        self.assertEqual(peak_response_idx, (exemplar_row, exemplar_col, 0))


if __name__ == '__main__':
    unittest.main()

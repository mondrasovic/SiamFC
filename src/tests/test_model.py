import unittest

import torch

from sot.model import SiamFC


class TestSiamFCModel(unittest.TestCase):
    def setUp(self) -> None:
        self.model = SiamFC()
    
    def test_feature_extraction_exemplar_shape(self):
        exemplar_img = torch.ones((1, 3, 127, 127))
        exemplar_emb = self.model.extract_visual_features(exemplar_img)

        self.assertEqual(list(exemplar_emb.shape), [1, 256, 6, 6])
    
    def test_feature_extraction_instance_shape(self):
        exemplar_img = torch.ones((1, 3, 255, 255))
        exemplar_emb = self.model.extract_visual_features(exemplar_img)

        self.assertEqual(list(exemplar_emb.shape), [1, 256, 22, 22])
    
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
        with self.assertRaises(ValueError):
            exemplar_emb = torch.ones((1, 1, 1))
            instance_emb = torch.ones(1, 10, 5, 5)
            self.model.cross_corr(exemplar_emb, instance_emb)


if __name__ == '__main__':
    unittest.main()

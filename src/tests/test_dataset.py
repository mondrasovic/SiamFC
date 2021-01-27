import unittest

from sot.dataset import ImageNetVideoDataset


class TesImageNetVideoDataset(unittest.TestCase):
    def test_invalid_subset(self):
        with self.assertRaises(AssertionError):
            ImageNetVideoDataset('some_path', 'invalid_subset')


if __name__ == '__main__':
    unittest.main()

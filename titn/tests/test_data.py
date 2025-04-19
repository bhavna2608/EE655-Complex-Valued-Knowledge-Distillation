import unittest
import torch
from data import build_dataset
from data.augmentations import CutMix, AutoAugment

class TestData(unittest.TestCase):
    def setUp(self):
        class Config:
            dataset = 'cifar10'
            data_dir = './data'
            use_autoaugment = True
            use_cutmix = True
            cutmix_alpha = 0.5
            img_size = 32
        
        self.config = Config()
    
    def test_dataset_loading(self):
        train_set, test_set = build_dataset(self.config)
        self.assertEqual(len(train_set), 50000)
        self.assertEqual(len(test_set), 10000)
        
        # Test CutMix
        img, target = train_set[0]
        if isinstance(target, tuple):
            self.assertEqual(len(target), 3)  # (label1, label2, lam)
        
        # Test normal sample
        img, target = test_set[0]
        self.assertTrue(isinstance(target, int))
    
    def test_cutmix(self):
        cutmix = CutMix(alpha=1.0)
        img1 = torch.randn(3, 32, 32)
        img2 = torch.randn(3, 32, 32)
        mixed_img, (label1, label2, lam) = cutmix(img1, img2, 0, 1)
        self.assertEqual(mixed_img.shape, img1.shape)
        self.assertTrue(0 <= lam <= 1)
    
    def test_autoaugment(self):
        aug = AutoAugment()
        img = torch.randn(3, 32, 32)
        aug_img = aug(img)
        self.assertEqual(aug_img.shape, img.shape)

if __name__ == '__main__':
    unittest.main()

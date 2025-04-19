import unittest
import torch
from models import TITN
from models.inner_transformer import InnerTransformerBlock
from models.outer_transformer import OuterTransformerBlock

class TestTITN(unittest.TestCase):
    def setUp(self):
        class Config:
            img_size = 32
            outer_patch = 8
            inner_patch = 4
            pixel_patch = 2
            outer_dim = 192
            inner_dim = 192
            depth = 2  # Shallow for testing
            num_heads = 8
            num_classes = 10
        
        self.config = Config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_inner_block(self):
        block = InnerTransformerBlock(self.config).to(self.device)
        x = torch.randn(2, 16, self.config.inner_dim).to(self.device)
        out = block(x)
        self.assertEqual(out.shape, x.shape)
    
    def test_outer_block(self):
        block = OuterTransformerBlock(self.config).to(self.device)
        x = torch.randn(2, 18, self.config.outer_dim).to(self.device)
        inner = torch.randn(2, 4, self.config.inner_dim).to(self.device)
        out = block(x, inner)
        self.assertEqual(out.shape, x.shape)
    
    def test_titn_forward(self):
        model = TITN(self.config).to(self.device)
        x = torch.randn(2, 3, 32, 32).to(self.device)
        cls_out, dist_out = model(x)
        self.assertEqual(cls_out.shape, (2, self.config.num_classes))
        self.assertEqual(dist_out.shape, (2, self.config.num_classes))

if __name__ == '__main__':
    unittest.main()
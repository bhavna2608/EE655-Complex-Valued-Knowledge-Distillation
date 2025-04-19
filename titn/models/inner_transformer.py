import torch
import torch.nn as nn
from .layers import Attention, MLPBlock

class InnerTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.inner_dim)
        self.attn = Attention(
            dim=config.inner_dim,
            num_heads=config.num_heads,
            qkv_bias=True,
            attn_drop=0.0,
            proj_drop=0.0
        )
        self.norm2 = nn.LayerNorm(config.inner_dim)
        self.mlp = MLPBlock(
            in_features=config.inner_dim,
            hidden_features=config.inner_dim * 4,
            act_layer=nn.GELU,
            drop=0.0
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
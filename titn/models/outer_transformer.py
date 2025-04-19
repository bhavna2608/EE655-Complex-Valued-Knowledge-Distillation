import torch
import torch.nn as nn
from .layers import Attention, MLPBlock

class OuterTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.outer_dim)
        self.attn = Attention(
            dim=config.outer_dim,
            num_heads=config.num_heads,
            qkv_bias=True,
            attn_drop=0.0,
            proj_drop=0.0
        )
        self.norm2 = nn.LayerNorm(config.outer_dim)
        self.mlp = MLPBlock(
            in_features=config.outer_dim,
            hidden_features=config.outer_dim * 4,
            act_layer=nn.GELU,
            drop=0.0
        )
        self.norm3 = nn.LayerNorm(config.outer_dim)
        self.proj = nn.Linear(config.inner_dim, config.outer_dim)
    
    def forward(self, x, inner_patches):
        # Cross-attention between outer and inner patches
        inner_proj = self.proj(self.norm3(inner_patches))
        x = x + self.attn(self.norm1(x), inner_proj)
        x = x + self.mlp(self.norm2(x))
        return x
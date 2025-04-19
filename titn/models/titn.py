import torch
import torch.nn as nn
from .inner_transformer import InnerTransformerBlock
from .outer_transformer import OuterTransformerBlock

class TITN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Patch embedding layers
        self.outer_patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=config["outer_dim"],
            kernel_size=config["outer_patch"],
            stride=config["outer_patch"]
        )
        
        self.inner_patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=config["inner_dim"],
            kernel_size=config["inner_patch"],
            stride=config["inner_patch"]
        )
        
        # Class and distillation tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config["outer_dim"]))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, config["outer_dim"]))
        
        # Position embeddings
        num_outer_patches = (config["img_size"] // config["outer_patch"]) ** 2
        self.outer_pos_embed = nn.Parameter(
            torch.zeros(1, num_outer_patches + 2, config["outer_dim"]))
        
        num_inner_patches = (config["inner_patch"] // config["pixel_patch"]) ** 2
        self.inner_pos_embed = nn.Parameter(
            torch.zeros(1, num_inner_patches, config["inner_dim"]))
        
        # Transformer blocks
        self.inner_blocks = nn.ModuleList([
            InnerTransformerBlock(config) for _ in range(config["depth"])
        ])
        self.outer_blocks = nn.ModuleList([
            OuterTransformerBlock(config) for _ in range(config["depth"])
        ])
        
        # Normalization and heads
        self.norm = nn.LayerNorm(config["outer_dim"])
        self.head = nn.Linear(config["outer_dim"], config["num_classes"])
        self.dist_head = nn.Linear(config["outer_dim"], config["num_classes"])
        
        self.init_weights()
    
    def init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.trunc_normal_(self.outer_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.inner_pos_embed, std=0.02)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.dist_head.weight)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Outer patch processing
        outer_patches = self.outer_patch_embed(x)
        outer_patches = outer_patches.flatten(2).transpose(1, 2)
        
        # Add tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_tokens = self.dist_token.expand(B, -1, -1)
        outer_patches = torch.cat((cls_tokens, outer_patches, dist_tokens), dim=1)
        outer_patches = outer_patches + self.outer_pos_embed
        
        # Inner patch processing
        inner_patches = self.inner_patch_embed(x)
        inner_patches = inner_patches.flatten(2).transpose(1, 2)
        inner_patches = inner_patches + self.inner_pos_embed
        
        # Transformer processing
        for inner_block, outer_block in zip(self.inner_blocks, self.outer_blocks):
            inner_patches = inner_block(inner_patches)
            outer_patches = outer_block(outer_patches, inner_patches)
        
        # Final processing
        outer_patches = self.norm(outer_patches)
        cls_out = outer_patches[:, 0]  # Class token
        dist_out = outer_patches[:, -1]  # Distillation token
        
        return self.head(cls_out), self.dist_head(dist_out)
import torch
import torch.nn as nn
from titn.models.layers import Attention, MLPBlock

class ViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        print(config)

        # Patch embedding
        self.patch_embed = nn.Conv2d(
        3, int(config['outer_dim']), 
        kernel_size=int(config['outer_patch']), 
        stride=int(config['outer_patch'])
)
        
        # Class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config["outer_dim"]))
        num_patches = (config["img_size"]// config["outer_patch"]) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config["outer_dim"]))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(config["outer_dim"]),
                Attention(config["outer_dim"], num_heads=config["num_heads"]),
                nn.LayerNorm(config["outer_dim"]),
                MLPBlock(config["outer_dim"], config["outer_dim"] * 4)
            ) for _ in range(config["depth"])
        ])
        
        # Head
        self.norm = nn.LayerNorm(config["outer_dim"])
        self.head = nn.Linear(config["outer_dim"], config["num_classes"])
        
        self.init_weights()
    
    def init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.zeros_(self.head.weight)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, C, H, W]
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        
        # Add class token and position embedding
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        # Classification head
        x = self.norm(x[:, 0])
        return self.head(x)
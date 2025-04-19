import torch
import torch.nn as nn
from titn.models.complex_layers import ComplexLinear, ComplexAttention

class ComplexTITN(nn.Module):
    def __init__(self, config):  # fixed typo: _init_ → __init__
        super().__init__()       # fixed typo: _init_ → __init__
        self.config = config

        # Complex patch embedding
        self.outer_patch_embed = nn.Sequential(
            nn.Conv2d(3, config["outer_dim"] * 2, kernel_size=config["outer_patch"], stride=config["outer_patch"]),
            nn.LayerNorm(config["outer_dim"] * 2),
            nn.GELU()
        )

        # Complex tokens and positions
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["outer_dim"], 2))
        self.pos_embed = nn.Parameter(torch.randn(1, (config["img_size"] // config["outer_patch"])**2 + 1, config["outer_dim"], 2))

        # Complex transformer blocks
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(config["outer_dim"] * 2),
                ComplexAttention(config["outer_dim"]),
                nn.LayerNorm(config["outer_dim"] * 2),
                ComplexLinear(config["outer_dim"], config["outer_dim"] * 4),
                ComplexLinear(config["outer_dim"] * 4, config["outer_dim"])
            ]) for _ in range(config["depth"])
        ])

        # Complex head
        self.norm = nn.LayerNorm(config["outer_dim"] * 2)
        self.head = ComplexLinear(config["outer_dim"], config["num_classes"])

    def complex_gelu(self, x):
        # Apply GELU separately to real and imaginary parts
        return torch.stack([
            nn.functional.gelu(x[..., 0]),
            nn.functional.gelu(x[..., 1])
        ], dim=-1)

    def forward(self, x):
        B = x.shape[0]

        # Complex embedding
        x = self.outer_patch_embed(x)  # [B, C*2, H, W]
        print(f"After outer_patch_embed: {x.shape}")
        x = x.reshape(B, -1, self.config["outer_dim"], 2)  # [B, N, C, 2]
        print(f"After reshape: {x.shape}")

        # Add cls token and position
        cls_tokens = self.cls_token.expand(B, -1, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        print(f"After adding cls token and position: {x.shape}")

        # Complex transformer
        for ln1, attn, ln2, fc1, fc2 in self.blocks:
            x_res = x
            x = ln1(x.reshape(B, -1, self.config["outer_dim"] * 2)).reshape(B, -1, self.config["outer_dim"], 2)
            x = attn(x) + x_res

            x_res = x
            x = ln2(x.reshape(B, -1, self.config["outer_dim"] * 2)).reshape(B, -1, self.config["outer_dim"], 2)
            x = fc2(self.complex_gelu(fc1(x))) + x_res  # ✅ GELU applied correctly
            print(f"After FC block: {x.shape}")

        # Complex classification
        x = self.norm(x.reshape(B, -1, self.config["outer_dim"] * 2)).reshape(B, -1, self.config["outer_dim"], 2)
        print(f"After final norm: {x.shape}")

        return self.head(x[:, 0])  # cls token
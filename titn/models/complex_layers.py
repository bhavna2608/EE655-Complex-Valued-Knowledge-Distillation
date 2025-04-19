import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight_real = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_imag = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_real = nn.Parameter(torch.Tensor(out_features))
        self.bias_imag = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_real, a=sqrt(5))
        nn.init.kaiming_uniform_(self.weight_imag, a=sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_real)
        bound = 1 / sqrt(fan_in)
        nn.init.uniform_(self.bias_real, -bound, bound)
        nn.init.uniform_(self.bias_imag, -bound, bound)

    def forward(self, x):
        x_real, x_imag = x[..., 0], x[..., 1]
        out_real = F.linear(x_real, self.weight_real, self.bias_real) - \
                   F.linear(x_imag, self.weight_imag, self.bias_imag)
        out_imag = F.linear(x_real, self.weight_imag, self.bias_imag) + \
                   F.linear(x_imag, self.weight_real, self.bias_real)
        return torch.stack([out_real, out_imag], dim=-1)

class ComplexAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1 / sqrt(self.head_dim)
        
        self.to_qkv = ComplexLinear(dim, dim * 3)
        self.to_out = ComplexLinear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, N, self.num_heads, self.head_dim, 2), qkv)
        
        attn = torch.einsum('bnhdc,bmhdc->bhnm', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = torch.einsum('bhnm,bmhdc->bnhdc', attn, v)
        out = out.reshape(B, N, C, 2)
        return self.to_out(out)
import torch
import torch.nn as nn
from utils.attn import SelfAttn


class LatentEncoder(nn.Module):
    def __init__(self, D: int, num_layers: int = 20, num_heads: int = 8):
        super().__init__()
        self.layers = nn.ModuleList([
            SelfAttn(D=D, num_heads=num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (L, D) -> (L, D)
        for layer in self.layers:
            x = layer(x)
        return x


class LatentDecoder(nn.Module):
    def __init__(self, D: int, num_layers: int = 4, num_heads: int = 8):
        super().__init__()
        self.layers = nn.ModuleList([
            SelfAttn(D=D, num_heads=num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (L, D) -> (L, D)
        for layer in self.layers:
            x = layer(x)
        return x

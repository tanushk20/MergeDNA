import torch
import torch.nn as nn
from utils.attn import LocalWindowSelfAttn


class LocalEncoder(nn.Module):
    def __init__(self, D: int, K: int, num_layers: int = 4, num_heads: int = 8):
        super().__init__()
        self.layers = nn.ModuleList([
            LocalWindowSelfAttn(D=D, K=K, num_heads=num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, D) -> (N, D)
        for layer in self.layers:
            x = layer(x)
        return x


class LocalDecoder(nn.Module):
    def __init__(self, D: int, K: int, num_layers: int = 2, num_heads: int = 8):
        super().__init__()
        self.layers = nn.ModuleList([
            LocalWindowSelfAttn(D=D, K=K, num_heads=num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, D) -> (N, D)
        for layer in self.layers:
            x = layer(x)
        return x

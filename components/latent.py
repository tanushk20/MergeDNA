from typing import Optional, Tuple

import torch
import torch.nn as nn

from utils.attn import SelfAttn, unmerge_source


class LatentEncoder(nn.Module):
    def __init__(self, D: int, num_layers: int = 20, reduce_by: int = 0):
        super().__init__()
        self.reduce_by = reduce_by
        self.layers = nn.ModuleList([SelfAttn(D=D) for _ in range(num_layers)])

    def forward(
        self, x: torch.Tensor, tome: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # x: (B, L, D) -> (B, L', D)
        source = None
        for layer in self.layers:
            x, source = layer(x, source, tome=tome, reduce_by=self.reduce_by)
        return x, source


class LatentDecoder(nn.Module):
    def __init__(self, D: int, num_layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([SelfAttn(D=D) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, source: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, L', D) -> (B, L, D)
        if source is not None:
            x = unmerge_source(x, source)
        for layer in self.layers:
            x, _ = layer(x)
        return x

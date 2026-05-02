from typing import Optional, Tuple

import torch
import torch.nn as nn

from utils.attn import LocalWindowSelfAttn, unmerge_source


class LocalEncoder(nn.Module):
    def __init__(self, D: int, K: int, num_layers: int = 4, reduce_by: int = 0):
        super().__init__()
        self.K = K
        self.reduce_by = reduce_by
        self.input_projector = nn.Linear(4, D)
        self.layers = nn.ModuleList([LocalWindowSelfAttn(D=D) for _ in range(num_layers)])

    def forward(
        self, x: torch.Tensor, source: Optional[torch.Tensor] = None,
        tome: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # x: (B, N, 4) -> (B, N', D)
        x = self.input_projector(x)
        B, N, D = x.shape
        num_windows = N // self.K

        for layer in self.layers:
            B_cur, N_cur, _ = x.shape
            K_cur = N_cur // num_windows

            x_win = x.view(B_cur * num_windows, K_cur, D)
            source_win = (
                source.view(B_cur * num_windows, K_cur, -1)
                if source is not None else None
            )

            x_win, source_win = layer(x_win, source_win, tome=tome, reduce_by=self.reduce_by)

            K_new = x_win.shape[1]
            x = x_win.view(B_cur, num_windows * K_new, D)
            if source_win is not None:
                source = source_win.view(B_cur, num_windows * K_new, -1)

        return x, source


class LocalDecoder(nn.Module):
    def __init__(self, D: int, K: int, num_layers: int = 2):
        super().__init__()
        self.K = K
        self.layers = nn.ModuleList([LocalWindowSelfAttn(D=D) for _ in range(num_layers)])
        self.output_projector = nn.Linear(D, 4)

    def forward(self, x: torch.Tensor, source: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, L', D), source: (B, L', N) -> (B, N, 4)
        if source is not None:
            x = unmerge_source(x, source)   # (B, N, D)

        B, N, D = x.shape
        num_windows = N // self.K
        x_win = x.view(B * num_windows, self.K, D)
        for layer in self.layers:
            x_win, _ = layer(x_win)
        return self.output_projector(x_win.view(B, N, D))

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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # x: (B, N, 4) -> (B, N', D)
        x = self.input_projector(x)
        B, N, D = x.shape
        num_windows = N // self.K
        size = None

        for layer in self.layers:
            B_cur, N_cur, _ = x.shape
            K_cur = N_cur // num_windows

            x_win = x.view(B_cur * num_windows, K_cur, D)
            source_win = (
                source.view(B_cur * num_windows, K_cur, -1)
                if source is not None else None
            )
            size_win = (
                size.view(B_cur * num_windows, K_cur, 1)
                if size is not None else None
            )

            x_win, source_win, size_win = layer(x_win, source_win, tome=tome, reduce_by=self.reduce_by, size=size_win)

            K_new = x_win.shape[1]
            x = x_win.view(B_cur, num_windows * K_new, D)
            if source_win is not None:
                source = source_win.view(B_cur, num_windows * K_new, -1)
            if size_win is not None:
                size = size_win.view(B_cur, num_windows * K_new, 1)

        # Expand window-local source (B, L', K_init) -> global (B, L', N)
        # so each merged token knows which of the N original tokens it covers.
        if source is not None:
            L_prime = source.shape[1]
            K_final = L_prime // num_windows          # tokens per window after all merges

            k_idx = torch.arange(self.K, device=x.device)           # (K_init,)
            m_idx = torch.arange(L_prime, device=x.device)          # (L',)
            win_idx = m_idx // K_final                               # (L',)
            global_col = (win_idx.unsqueeze(1) * self.K             # (L', K_init)
                          + k_idx.unsqueeze(0))
            global_col = global_col.unsqueeze(0).expand(B, -1, -1)  # (B, L', K_init)

            global_source = torch.zeros(B, L_prime, N, device=x.device)
            global_source.scatter_(-1, global_col, source)           # (B, L', N)
            source = global_source

        return x, source, size


class LocalDecoder(nn.Module):
    def __init__(self, D: int, K: int, num_layers: int = 2):
        super().__init__()
        self.K = K
        self.layers = nn.ModuleList([LocalWindowSelfAttn(D=D) for _ in range(num_layers)])
        self.output_projector = nn.Linear(D, 4)

    def forward(self, x: torch.Tensor, source: Optional[torch.Tensor] = None, size: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, L', D), source: (B, L', N) -> (B, N, 4)
        if source is not None:
            x = unmerge_source(x, source)   # (B, N, D)

        B, N, D = x.shape
        num_windows = N // self.K
        x_win = x.view(B * num_windows, self.K, D)
        for layer in self.layers:
            x_win, _, _ = layer(x_win)
        return self.output_projector(x_win.view(B, N, D))

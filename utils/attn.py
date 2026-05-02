import torch
import torch.nn as nn


class LocalWindowSelfAttn(nn.Module):
    def __init__(self, D: int, K: int, num_heads: int = 8):
        super().__init__()
        self.K = K
        self.attn = nn.MultiheadAttention(D, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(D)
        self.ff = nn.Sequential(
            nn.Linear(D, 4 * D),
            nn.GELU(),
            nn.Linear(4 * D, D),
        )
        self.norm2 = nn.LayerNorm(D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        B, N, D = x.shape
        K = self.K
        if N % K != 0:
            raise ValueError(f"Sequence length N={N} must be divisible by window size K={K}")

        num_windows = N // K
        # reshape into windows: (B * num_windows, K, D)
        x_win = x.view(B * num_windows, K, D)

        attn_out, _ = self.attn(x_win, x_win, x_win)
        x_win = self.norm(x_win + attn_out)
        x_win = self.norm2(x_win + self.ff(x_win))

        return x_win.view(B, N, D)


class SelfAttn(nn.Module):
    def __init__(self, D: int, num_heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(D, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(D)
        self.ff = nn.Sequential(
            nn.Linear(D, 4 * D),
            nn.GELU(),
            nn.Linear(4 * D, D),
        )
        self.norm2 = nn.LayerNorm(D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x

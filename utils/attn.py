from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.tome import bipartite_soft_matching, merge_wavg, merge_source, unmerge_source


class LocalWindowSelfAttn(nn.Module):
    """
    Self-attention + FFN for pre-windowed input (B*nw, K_cur, D).
    Windowing and source reshaping are handled by LocalEncoder/LocalDecoder.
    """
    def __init__(self, D: int):
        super().__init__()
        self.q_proj = nn.Linear(D, D)
        self.k_proj = nn.Linear(D, D)
        self.v_proj = nn.Linear(D, D)
        self.out_proj = nn.Linear(D, D)
        self.norm = nn.LayerNorm(D)
        self.ff = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
        self.norm2 = nn.LayerNorm(D)
        self.metric_proj = nn.Linear(D, D, bias=False)

    def forward(
        self, x: torch.Tensor, source: Optional[torch.Tensor] = None,
        tome: bool = False, reduce_by: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # x: (B*nw, K_cur, D)
        metric = self.metric_proj(x)                # compute before attention
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        scale = Q.shape[-1] ** -0.5
        attn_out = F.softmax((Q @ K.transpose(-1, -2)) * scale, dim=-1) @ V
        x = self.norm(x + self.out_proj(attn_out))

        if tome:
            merge, _ = bipartite_soft_matching(metric, r=reduce_by)
            source = merge_source(merge, x, source)
            x, _ = merge_wavg(merge, x)

        x = self.norm2(x + self.ff(x))
        return x, source


class SelfAttn(nn.Module):
    """
    Full self-attention + FFN for (B, N, D).
    """
    def __init__(self, D: int):
        super().__init__()
        self.q_proj = nn.Linear(D, D)
        self.k_proj = nn.Linear(D, D)
        self.v_proj = nn.Linear(D, D)
        self.out_proj = nn.Linear(D, D)
        self.norm = nn.LayerNorm(D)
        self.ff = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
        self.norm2 = nn.LayerNorm(D)
        self.metric_proj = nn.Linear(D, D, bias=False)

    def forward(
        self, x: torch.Tensor, source: Optional[torch.Tensor] = None,
        tome: bool = False, reduce_by: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # x: (B, N, D)
        metric = self.metric_proj(x)                # compute before attention
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        scale = Q.shape[-1] ** -0.5
        attn_out = F.softmax((Q @ K.transpose(-1, -2)) * scale, dim=-1) @ V
        x = self.norm(x + self.out_proj(attn_out))

        if tome:
            merge, _ = bipartite_soft_matching(metric, r=reduce_by)
            source = merge_source(merge, x, source)
            x, _ = merge_wavg(merge, x)

        x = self.norm2(x + self.ff(x))
        return x, source

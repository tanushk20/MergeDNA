from typing import Callable, Tuple

import torch


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe bipartite soft matching.

    Args:
        metric: Key tensors of shape (B, K, D) used to compute token similarity.
        r:      Number of tokens to merge (max 50% of tokens).

    Returns:
        merge:   fn (B, K, D) -> (B, K-r, D)  — merges r similar token pairs
        unmerge: fn (B, K-r, D) -> (B, K, D)  — reconstructs original length
    """
    t = metric.shape[1]
    r = min(r, t // 2)

    if r <= 0:
        def do_nothing(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
            return x
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)

        # split tokens into even (a) and odd (b) sets
        a, b = metric[..., ::2, :], metric[..., 1::2, :]

        # similarity scores between every a-b pair
        scores = a @ b.transpose(-1, -2)       # (B, K//2, K//2)

        # for each a-token, find its most similar b-token
        node_max, node_idx = scores.max(dim=-1)     # (B, K//2)

        # rank a-tokens by similarity (highest first = best merge candidates)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]  # (B, K//2, 1)

        unm_idx = edge_idx[..., r:, :]     # (B, K//2 - r, 1) — tokens to keep
        src_idx = edge_idx[..., :r, :]     # (B, r, 1)         — tokens to merge
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)  # (B, r, 1)

    def merge(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        # x: (B, K, D)
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        B, t1, D = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(B, t1 - r, D))
        src = src.gather(dim=-2, index=src_idx.expand(B, r, D))
        dst = dst.scatter_reduce(-2, dst_idx.expand(B, r, D), src, reduce=mode)
        return torch.cat([unm, dst], dim=1)     # (B, K-r, D)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        # x: (B, K-r, D)
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        B, _, D = unm.shape
        src = dst.gather(dim=-2, index=dst_idx.expand(B, r, D))
        out = torch.zeros(B, metric.shape[1], D, device=x.device, dtype=x.dtype)
        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(B, unm_len, D), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(B, r, D), src=src)
        return out  # (B, K, D)

    return merge, unmerge


def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Weighted average merge based on token size.
    Returns the merged tensor and updated token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")
    x = x / size
    return x, size


def merge_source(
    merge: Callable, x: torch.Tensor, source: torch.Tensor = None
) -> torch.Tensor:
    """
    Tracks which original tokens contributed to each merged token.
    source is an adjacency matrix (B, T, T) between original tokens and merged groups.
    """
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

    source = merge(source, mode="amax")
    return source


def unmerge_source(x: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """
    Unmerges tokens using the source matrix.

    x:      (B, L', D)  — merged token embeddings
    source: (B, L', N)  — which original tokens each merged token covers
    returns (B, N, D)   — each original token gets its merged token's embedding
    """
    return source.transpose(-1, -2) @ x

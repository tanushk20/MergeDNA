import torch
import torch.nn as nn
import torch.nn.functional as F


class LossManager(nn.Module):
    def __init__(self):
        super().__init__()

    def loss(
        self,
        batch: torch.Tensor,
        local_encoder: nn.Module,
        latent_encoder: nn.Module,
        latent_decoder: nn.Module,
        local_decoder: nn.Module,
    ) -> torch.Tensor:
        # forward pass: (B, L, 4) -> (B, L, D) -> ... -> (B, L, 4)
        x = local_encoder(batch)
        x = latent_encoder(x)
        x = latent_decoder(x)
        logits = local_decoder(x)   # (B, L, 4)

        # targets: one-hot -> class indices (B, L)
        targets = batch.argmax(dim=-1)

        loss = F.cross_entropy(logits.view(-1, 4), targets.view(-1))
        return loss

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LossManager(nn.Module):
    def __init__(self, weight_mtr: float = 1.0, weight_mtr_latent: float = 1.0, weight_amtm: float = 1.0):
        super().__init__()
        self.weight_mtr = weight_mtr
        self.weight_mtr_latent = weight_mtr_latent
        self.weight_amtm = weight_amtm

    def compute_loss_mtr(
        self,
        batch: torch.Tensor,
        local_encoder: nn.Module,
        latent_encoder: nn.Module,
        latent_decoder: nn.Module,
        local_decoder: nn.Module,
    ) -> torch.Tensor:
        x, local_source, local_size = local_encoder(batch, tome=True)
        x, _, _ = latent_encoder(x, tome = False)
        x = latent_decoder(x)
        logits = local_decoder(x, local_source)     # (B, N, 4)

        targets = batch.argmax(dim=-1)
        valid_mask = batch.sum(dim=-1) > 0

        return F.cross_entropy(
            logits[valid_mask],
            targets[valid_mask],
        )

    def compute_loss_mtr_latent(
        self,
        batch: torch.Tensor,
        local_encoder: nn.Module,
        latent_encoder: nn.Module,
        latent_decoder: nn.Module,
        local_decoder: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, local_source, local_size = local_encoder(batch, tome=True)
        x, local_source, local_size = x.detach(), local_source.detach(), local_size.detach()

        x, latent_source, latent_size = latent_encoder(x, tome=True, source=None, size=local_size)
        x = latent_decoder(x, latent_source)        # unmerge latent: K' -> L
        logits = local_decoder(x, local_source)     # unmerge local: L -> N

        targets = batch.argmax(dim=-1)
        valid_mask = batch.sum(dim=-1) > 0

        loss = F.cross_entropy(
            logits[valid_mask],
            targets[valid_mask],
        )
        return loss, latent_source, local_source

    def sample_masks(
        self,
        latent_source: torch.Tensor,
        local_source: torch.Tensor,
        num_masked_tokens: int,
    ) -> torch.Tensor:
        # latent_source: (B, K', L) — which local tokens each latent token covers
        # local_source:  (B, L, N) — which original tokens each local token covers

        g = latent_source.sum(dim=-1)                                            # (B, K')
        prob = torch.einsum('bkl,bk->bl', latent_source, 1.0 / g ** 2)          # (B, L)
        prob = prob / prob.sum(dim=-1, keepdim=True)

        sampled = torch.multinomial(prob, num_samples=num_masked_tokens, replacement=False)      # (B, K)
        M_L = torch.zeros_like(prob).scatter_(1, sampled, 1.0)                  # (B, L)

        M_N = (local_source.transpose(-1, -2) @ M_L.unsqueeze(-1)).squeeze(-1)  # (B, N)
        return (M_N > 0).float()

    def compute_loss_adaptive(
        self,
        batch: torch.Tensor,
        M_N: torch.Tensor,
        local_encoder: nn.Module,
        latent_encoder: nn.Module,
        latent_decoder: nn.Module,
        local_decoder: nn.Module,
    ) -> torch.Tensor:
        # zero out masked positions in input: X * (1 - M_N)
        masked_batch = batch * (1.0 - M_N).unsqueeze(-1)   # (B, N, 4)

        x, local_source, local_size = local_encoder(masked_batch, tome=True)
        x, _ , _= latent_encoder(x, tome = False)
        x = latent_decoder(x)
        logits = local_decoder(x, local_source)             # (B, N, 4)

        # loss only on masked positions, averaged over K masked tokens
        targets = batch.argmax(dim=-1)                      # (B, N)
        valid_mask = batch.sum(dim=-1) > 0                  # (B, N)

        log_probs = F.log_softmax(logits, dim=-1)           # (B, N, 4)
        correct = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B, N)

        mask = M_N * valid_mask.float()
        return -(correct * mask).sum() / mask.sum().clamp_min(1.0)

    def loss(
        self,
        batch: torch.Tensor,
        local_encoder: nn.Module,
        latent_encoder: nn.Module,
        latent_decoder: nn.Module,
        local_decoder: nn.Module,
        num_masked_tokens: int,
    ) -> Tuple[torch.Tensor, dict]:
        loss_mtr = self.compute_loss_mtr(
            batch, local_encoder, latent_encoder, latent_decoder, local_decoder,
        )
        loss_mtr_latent, latent_source, local_source = self.compute_loss_mtr_latent(
            batch, local_encoder, latent_encoder, latent_decoder, local_decoder,
        )
        M_N = self.sample_masks(latent_source.detach(), local_source.detach(), num_masked_tokens)
        loss_amtm = self.compute_loss_adaptive(
            batch, M_N, local_encoder, latent_encoder, latent_decoder, local_decoder,
        )
        total = (
            self.weight_mtr * loss_mtr
            + self.weight_mtr_latent * loss_mtr_latent
            + self.weight_amtm * loss_amtm
        )
        breakdown = {
            "mtr": loss_mtr.item(),
            "mtr_latent": loss_mtr_latent.item(),
            "amtm": loss_amtm.item(),
            "total": total.item(),
        }
        return total, breakdown

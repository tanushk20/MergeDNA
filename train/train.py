import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


class Trainer:
    def __init__(
        self,
        dataset: Dataset,
        projector: nn.Module,
        local_encoder: nn.Module,
        local_decoder: nn.Module,
        latent_encoder: nn.Module,
        latent_decoder: nn.Module,
        optimizer: torch.optim.Optimizer,
        val_split: float = 0.1,
        batch_size: int = 8,
        num_workers: int = 4,
        device: str = "cpu",
    ):
        self.projector = projector
        self.local_encoder = local_encoder
        self.local_decoder = local_decoder
        self.latent_encoder = latent_encoder
        self.latent_decoder = latent_decoder
        self.optimizer = optimizer
        self.device = device

        self.train_loader, self.val_loader = self._split_dataset(
            dataset, val_split, batch_size, num_workers
        )

    def _split_dataset(
        self, dataset: Dataset, val_split: float, batch_size: int, num_workers: int
    ):
        val_len = int(len(dataset) * val_split)
        train_len = len(dataset) - val_len
        train_set, val_set = random_split(dataset, [train_len, val_len])

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, val_loader

    def _forward(self, batch: torch.Tensor) -> torch.Tensor:
        # batch: (B, L, 4) -> (B, L, D)
        x = batch.to(self.device)
        x = self.projector(x)
        x = self.local_encoder(x)
        x = self.latent_encoder(x)
        x = self.latent_decoder(x)
        x = self.local_decoder(x)
        return x

    def train(self, num_epochs: int = 10):
        models = [self.projector, self.local_encoder, self.local_decoder,
                  self.latent_encoder, self.latent_decoder]
        for m in models:
            m.to(self.device).train()

        for epoch in range(num_epochs):
            for batch in self.train_loader:
                self.optimizer.zero_grad()
                out = self._forward(batch)
                # loss goes here
                # self.optimizer.step()

            print(f"Epoch {epoch + 1}/{num_epochs} done | "
                  f"train batches: {len(self.train_loader)} | "
                  f"val batches: {len(self.val_loader)}")

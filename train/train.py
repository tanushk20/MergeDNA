import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm


LOG_EVERY = 10


class Trainer:
    def __init__(
        self,
        dataset: Dataset,
        loss_manager: nn.Module,
        local_encoder: nn.Module,
        local_decoder: nn.Module,
        latent_encoder: nn.Module,
        latent_decoder: nn.Module,
        optimizer: torch.optim.Optimizer,
        K: int,
        val_split: float = 0.1,
        batch_size: int = 8,
        num_workers: int = 4,
        device: str = "cpu",
    ):
        self.loss_manager = loss_manager
        self.K = K
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

    def train(self, num_epochs: int = 10):
        models = [self.loss_manager, self.local_encoder, self.local_decoder,
                  self.latent_encoder, self.latent_decoder]
        for m in models:
            m.to(self.device).train()

        for epoch in range(num_epochs):
            running = {"mtr": 0.0, "mtr_latent": 0.0, "amtm": 0.0, "total": 0.0}

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)
            for step, batch in enumerate(pbar, 1):
                batch = batch.to(self.device)
                self.optimizer.zero_grad()

                loss, breakdown = self.loss_manager.loss(
                    batch,
                    self.local_encoder,
                    self.latent_encoder,
                    self.latent_decoder,
                    self.local_decoder,
                    K=self.K,
                )
                loss.backward()
                self.optimizer.step()

                for k, v in breakdown.items():
                    running[k] += v

                if step % LOG_EVERY == 0:
                    avg = {k: v / LOG_EVERY for k, v in running.items()}
                    pbar.write(
                        f"  step {step:5d} | "
                        f"total {avg['total']:.4f} | "
                        f"mtr {avg['mtr']:.4f} | "
                        f"mtr_latent {avg['mtr_latent']:.4f} | "
                        f"amtm {avg['amtm']:.4f}"
                    )
                    running = {k: 0.0 for k in running}

            epoch_avg = running["total"] / (len(self.train_loader) % LOG_EVERY or LOG_EVERY)
            pbar.write(f"Epoch {epoch + 1}/{num_epochs} done | avg total loss: {epoch_avg:.4f}")

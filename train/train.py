import math
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from tqdm import tqdm


LOG_EVERY = 10


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _worker_init_fn(worker_id: int):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _make_scheduler(optimizer, n_warmup: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < n_warmup:
            return step / max(1, n_warmup)
        progress = (step - n_warmup) / max(1, total_steps - n_warmup)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


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
        seed: int = 42,
        val_split: float = 0.1,
        batch_size: int = 8,
        num_workers: int = 4,
        device: str = "cpu",
        dummy: bool = False,
        grad_accum_steps: int = 1,
        n_warmup: int = 10000,
    ):
        self.loss_manager = loss_manager
        self.K = K
        self.local_encoder = local_encoder
        self.local_decoder = local_decoder
        self.latent_encoder = latent_encoder
        self.latent_decoder = latent_decoder
        self.optimizer = optimizer
        self.device = device
        self.dummy = dummy
        self.grad_accum_steps = grad_accum_steps
        self.n_warmup = n_warmup

        set_seed(seed)
        self.train_loader, self.val_loader = self._split_dataset(
            dataset, val_split, batch_size, num_workers, seed
        )

    def _split_dataset(
        self, dataset: Dataset, val_split: float, batch_size: int, num_workers: int, seed: int
    ):
        val_len = int(len(dataset) * val_split)
        train_len = len(dataset) - val_len
        train_set, val_set = random_split(
            dataset, [train_len, val_len],
            generator=torch.Generator().manual_seed(seed),
        )

        if self.dummy:
            train_set = Subset(train_set, range(min(64, len(train_set))))

        g = torch.Generator()
        g.manual_seed(seed)
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, worker_init_fn=_worker_init_fn, generator=g,
        )
        val_loader = DataLoader(
            val_set, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, worker_init_fn=_worker_init_fn,
        )
        return train_loader, val_loader

    def train(self, num_epochs: int = 10):
        models = [self.loss_manager, self.local_encoder, self.local_decoder,
                  self.latent_encoder, self.latent_decoder]
        for m in models:
            m.to(self.device).train()

        steps_per_epoch = math.ceil(len(self.train_loader) / self.grad_accum_steps)
        total_steps = num_epochs * steps_per_epoch
        scheduler = _make_scheduler(self.optimizer, self.n_warmup, total_steps)

        self.optimizer.zero_grad()
        for epoch in range(num_epochs):
            running = {"mtr": 0.0, "mtr_latent": 0.0, "amtm": 0.0, "total": 0.0}
            epoch_running = {"mtr": 0.0, "mtr_latent": 0.0, "amtm": 0.0, "total": 0.0}

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)
            for step, batch in enumerate(pbar, 1):
                batch = batch.to(self.device)

                loss, breakdown = self.loss_manager.loss(
                    batch,
                    self.local_encoder,
                    self.latent_encoder,
                    self.latent_decoder,
                    self.local_decoder,
                    K=self.K,
                )
                (loss / self.grad_accum_steps).backward()

                for k, v in breakdown.items():
                    running[k] += v
                    epoch_running[k] += v

                if step % self.grad_accum_steps == 0:
                    self.optimizer.step()
                    scheduler.step()
                    self.optimizer.zero_grad()

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

            # flush any remaining accumulated gradients at epoch end
            if len(self.train_loader) % self.grad_accum_steps != 0:
                self.optimizer.step()
                scheduler.step()
                self.optimizer.zero_grad()

            epoch_avg = {k: v / len(self.train_loader) for k, v in epoch_running.items()}
            pbar.write(
                f"Epoch {epoch + 1}/{num_epochs} done | "
                f"train total {epoch_avg['total']:.4f} | "
                f"mtr {epoch_avg['mtr']:.4f} | "
                f"mtr_latent {epoch_avg['mtr_latent']:.4f} | "
                f"amtm {epoch_avg['amtm']:.4f}"
            )

            torch.save({
                "epoch": epoch + 1,
                "local_encoder": self.local_encoder.state_dict(),
                "local_decoder": self.local_decoder.state_dict(),
                "latent_encoder": self.latent_encoder.state_dict(),
                "latent_decoder": self.latent_decoder.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, f"checkpoint_epoch{epoch + 1}.pt")

            # validation
            for m in models:
                m.eval()
            val_running = {"mtr": 0.0, "mtr_latent": 0.0, "amtm": 0.0, "total": 0.0}
            with torch.no_grad():
                for batch in tqdm(self.val_loader, desc="  val", leave=False):
                    batch = batch.to(self.device)
                    _, breakdown = self.loss_manager.loss(
                        batch,
                        self.local_encoder,
                        self.latent_encoder,
                        self.latent_decoder,
                        self.local_decoder,
                        K=self.K,
                    )
                    for k, v in breakdown.items():
                        val_running[k] += v
            val_avg = {k: v / len(self.val_loader) for k, v in val_running.items()}
            print(
                f"  val total {val_avg['total']:.4f} | "
                f"mtr {val_avg['mtr']:.4f} | "
                f"mtr_latent {val_avg['mtr_latent']:.4f} | "
                f"amtm {val_avg['amtm']:.4f}"
            )
            for m in models:
                m.train()
import torch

from dataloader.dataloader import DNADataset
from components.local import LocalEncoder, LocalDecoder
from components.latent import LatentEncoder, LatentDecoder
from loss.loss import LossManager
from train.train import Trainer

FASTA_PATH = "data/human_chr1_4096_filtered.fa"
D = 1024
K = 16


def load_components() -> dict:
    dataset = DNADataset(FASTA_PATH)

    local_encoder = LocalEncoder(D=D, K=K, num_layers=4)
    latent_encoder = LatentEncoder(D=D, num_layers=20)
    latent_decoder = LatentDecoder(D=D, num_layers=4)
    local_decoder = LocalDecoder(D=D, K=K, num_layers=2)
    loss_manager = LossManager()

    optimizer = torch.optim.AdamW(
        list(local_encoder.parameters()) +
        list(latent_encoder.parameters()) +
        list(latent_decoder.parameters()) +
        list(local_decoder.parameters()),
        lr=1e-4,
    )

    trainer = Trainer(
        dataset=dataset,
        loss_manager=loss_manager,
        local_encoder=local_encoder,
        local_decoder=local_decoder,
        latent_encoder=latent_encoder,
        latent_decoder=latent_decoder,
        optimizer=optimizer,
    )

    return {
        "dataset": dataset,
        "local_encoder": local_encoder,
        "latent_encoder": latent_encoder,
        "latent_decoder": latent_decoder,
        "local_decoder": local_decoder,
        "loss_manager": loss_manager,
        "optimizer": optimizer,
        "trainer": trainer,
    }


if __name__ == "__main__":
    components = load_components()
    components["trainer"].train()
    print("Model trained")

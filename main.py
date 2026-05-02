import argparse
import yaml
import torch

from dataloader.dataloader import DNADataset
from components.local import LocalEncoder, LocalDecoder
from components.latent import LatentEncoder, LatentDecoder
from loss.loss import LossManager
from train.train import Trainer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_components(cfg: dict) -> dict:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    dataset = DNADataset(data_cfg["fasta_path"])

    D = model_cfg["D"]
    K = model_cfg["K"]
    num_heads = model_cfg["num_heads"]

    local_encoder = LocalEncoder(D=D, K=K, num_layers=model_cfg["local_encoder_layers"], num_heads=num_heads)
    latent_encoder = LatentEncoder(D=D, num_layers=model_cfg["latent_encoder_layers"], num_heads=num_heads)
    latent_decoder = LatentDecoder(D=D, num_layers=model_cfg["latent_decoder_layers"], num_heads=num_heads)
    local_decoder = LocalDecoder(D=D, K=K, num_layers=model_cfg["local_decoder_layers"], num_heads=num_heads)
    loss_manager = LossManager()

    optimizer = torch.optim.AdamW(
        list(local_encoder.parameters()) +
        list(latent_encoder.parameters()) +
        list(latent_decoder.parameters()) +
        list(local_decoder.parameters()),
        lr=train_cfg["lr"],
    )

    trainer = Trainer(
        dataset=dataset,
        loss_manager=loss_manager,
        local_encoder=local_encoder,
        local_decoder=local_decoder,
        latent_encoder=latent_encoder,
        latent_decoder=latent_decoder,
        optimizer=optimizer,
        val_split=train_cfg["val_split"],
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
        device=train_cfg["device"],
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/TEMPLATE.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    components = load_components(cfg)
    components["trainer"].train(num_epochs=cfg["training"]["num_epochs"])
    print("Model trained")

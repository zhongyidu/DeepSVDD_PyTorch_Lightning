import os
import random
import numpy as np
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from utils.config import Config
from datasets import CIFAR10_Dataset
from networks import build_network, build_autoencoder
from lightning_modules.deep_svdd_module import DeepSVDDModule
from lightning_modules.autoencoder_module import AutoencoderModule
import yaml


def main():
    # Load YAML configuration
    config_path = os.path.join("config", "config.yaml")
    cfg = Config(config_path).settings

    # Set random seed
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create experiment directory
    xp_path = cfg["experiment"]["xp_path"]
    os.makedirs(xp_path, exist_ok=True)

    # Initialize dataset and data loaders
    data_cfg = cfg["data"]
    dataset = CIFAR10_Dataset(
        root=data_cfg["data_path"], normal_class=data_cfg["normal_class"]
    )
    train_loader, val_loader = dataset.loaders(
        batch_size=cfg["train"]["batch_size"], num_workers=0
    )

    # Build the network (feature extractor)
    net = build_network(cfg["network"]["net_name"])

    # Optional: Pretrain using an autoencoder
    if cfg["pretrain"]["enabled"]:
        autoencoder = build_autoencoder(cfg["network"]["net_name"])
        ae_module = AutoencoderModule(
            autoencoder,
            lr=cfg["pretrain"]["lr"],
            weight_decay=cfg["pretrain"]["weight_decay"],
            optimizer_name=cfg["pretrain"]["optimizer"],
            lr_milestones=cfg["pretrain"]["lr_milestones"],
        )
        early_stopping_callback = EarlyStopping(
            monitor="val_loss", mode="min", patience=50
        )
        ae_trainer = L.Trainer(
            max_epochs=cfg["pretrain"]["n_epochs"], callbacks=[early_stopping_callback]
        )
        ae_trainer.fit(ae_module, train_loader, val_loader)
        # Transfer encoder weights from autoencoder to the main network
        net_dict = net.state_dict()
        ae_dict = autoencoder.state_dict()
        pretrained_dict = {k: v for k, v in ae_dict.items() if k in net_dict}
        net_dict.update(pretrained_dict)
        net.load_state_dict(net_dict)

    # Initialize the Deep SVDD Lightning module
    deep_svdd_module = DeepSVDDModule(
        net=net,
        objective=cfg["objective"],
        nu=cfg["nu"],
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        optimizer_name=cfg["train"]["optimizer"],
        lr_milestones=cfg["train"]["lr_milestones"],
    )

    # Set up a checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=xp_path,
        filename="deep_svdd-{epoch:02d}-{train_loss:.2f}",
        save_top_k=1,
        monitor="train_loss",
        mode="min",
    )

    # Create the PyTorch Lightning Trainer and run training/testing
    trainer = L.Trainer(
        max_epochs=cfg["train"]["n_epochs"],
        default_root_dir=xp_path,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(deep_svdd_module, train_loader, val_loader)
    trainer.test(deep_svdd_module, val_loader)

    # Save the final model and configuration for reproducibility
    torch.save(deep_svdd_module.state_dict(), os.path.join(xp_path, "model.ckpt"))
    with open(os.path.join(xp_path, "config.yaml"), "w") as f:
        yaml.dump(cfg, f)


if __name__ == "__main__":
    main()

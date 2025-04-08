import torch
import torch.nn as nn
import lightning as L
from sklearn.metrics import roc_auc_score
import numpy as np
from typing import List


class AutoencoderModule(L.LightningModule):
    def __init__(
        self,
        autoencoder: nn.Module,
        lr: float = 0.001,
        weight_decay: float = 1e-6,
        optimizer_name: str = "adam",
        lr_milestones: List = [50],
        gamma: float = 0.1,
    ):
        """
        Lightning module for autoencoder training with an added StepLR scheduler.

        Args:
            autoencoder (nn.Module): The autoencoder network.
            lr (float): Learning rate.
            weight_decay (float): Weight decay for optimizer.
            optimizer_name (str): Optimizer type ('adam' or 'amsgrad').
            step_size (int): Period of learning rate decay.
            gamma (float): Multiplicative factor of learning rate decay.
        """
        super(AutoencoderModule, self).__init__()
        self.autoencoder = autoencoder
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.lr_milestones = lr_milestones
        self.gamma = gamma

    def forward(self, x):
        return self.autoencoder(x)

    def training_step(self, batch, batch_idx):
        inputs, _, _ = batch
        outputs = self.forward(inputs)
        loss = nn.functional.mse_loss(outputs, inputs)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, _, _ = batch
        outputs = self.forward(inputs)
        loss = nn.functional.mse_loss(outputs, inputs)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels, indices = batch
        outputs = self.forward(inputs)
        # Compute reconstruction error as sum of squared differences per sample
        scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
        loss = torch.mean(scores)
        self.log("test_loss", loss, prog_bar=True)
        return {
            "test_loss": loss,
            "scores": scores,
            "labels": labels,
            "indices": indices,
        }

    def test_epoch_end(self, outputs):
        # Aggregate scores and labels from all test batches
        all_scores = torch.cat([x["scores"] for x in outputs]).cpu().numpy()
        all_labels = torch.cat([x["labels"] for x in outputs]).cpu().numpy()
        avg_test_loss = np.mean([x["test_loss"].item() for x in outputs])
        auc = roc_auc_score(all_labels, all_scores)
        self.log("avg_test_loss", avg_test_loss, prog_bar=True)
        self.log("test_auc", auc, prog_bar=True)

    def configure_optimizers(self):
        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.autoencoder.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == "amsgrad":
            optimizer = torch.optim.Adam(
                self.autoencoder.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                amsgrad=True,
            )
        else:
            raise ValueError("Unsupported optimizer")
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.lr_milestones, gamma=self.gamma
        )
        return [optimizer], [scheduler]

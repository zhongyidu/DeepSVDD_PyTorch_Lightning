import torch
import torch.nn as nn
import lightning as L
import numpy as np
from sklearn.metrics import roc_auc_score
from typing import List


def get_radius(distances: np.ndarray, nu: float):
    """
    Computes the optimal radius R as the (1 - nu)-quantile of the square roots of the distances.
    """
    return np.quantile(np.sqrt(distances), 1 - nu)


class DeepSVDDModule(L.LightningModule):
    def __init__(
        self,
        net: nn.Module,
        objective: str = "one-class",
        nu: float = 0.1,
        lr: float = 0.001,
        weight_decay: float = 1e-6,
        warm_up_epochs: int = 10,
        optimizer_name: str = "adam",
        lr_milestones: List = [50],
        scheduler_gamma: float = 0.1,
    ):
        """
        Lightning module for Deep SVDD training.

        Args:
            net (nn.Module): The feature extractor network.
            objective (str): 'one-class' or 'soft-boundary'.
            nu (float): Hyperparameter nu (0 < nu <= 1).
            lr (float): Learning rate.
            weight_decay (float): Weight decay for optimizer.
            warm_up_epochs (int): Number of epochs to wait before updating the radius R.
            optimizer_name (str): Optimizer type ('adam' or 'amsgrad').
            scheduler_step_size (int): Step size for the LR scheduler.
            scheduler_gamma (float): Gamma factor for the LR scheduler.
        """
        super(DeepSVDDModule, self).__init__()
        self.net = net
        self.objective = objective
        self.nu = nu
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.warm_up_epochs = warm_up_epochs
        self.lr_milestones = lr_milestones
        self.scheduler_gamma = scheduler_gamma

        self.c = None  # hypersphere center (to be initialized before training)
        self.R = 0.0  # radius (used only for soft-boundary objective)

        self._train_epoch_dists = []
        self._test_outputs = []

    def forward(self, x):
        return self.net(x)

    def compute_loss(self, outputs):
        # Compute squared Euclidean distances from the center c.
        dist = torch.sum((outputs - self.c) ** 2, dim=1)
        if self.objective == "soft-boundary":
            # For soft-boundary, loss is R² plus scaled hinge loss on (distance - R²)
            scores = dist - self.R**2
            loss = self.R**2 + (1.0 / self.nu) * torch.mean(torch.clamp(scores, min=0))
        else:
            loss = torch.mean(dist)
        return loss, dist

    def on_train_start(self):
        """
        Automatically initialize the center c using the training dataloader if not already set.
        """
        if self.c is None:
            train_loader = self.trainer.train_dataloader
            self.init_center(train_loader)
            self.log("center_initialized", True)

    def training_step(self, batch, batch_idx):
        inputs, _, _ = batch
        outputs = self.forward(inputs)
        loss, dist = self.compute_loss(outputs)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        # Return distances for potential warm-up update later.
        return {"loss": loss, "dist": dist}

    def on_train_epoch_end(self):
        # For soft-boundary objective, update R if past warm-up epochs.
        # current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        if (
            self.objective == "soft-boundary"
            and self.current_epoch >= self.warm_up_epochs
        ):
            all_dists = torch.cat(self._train_epoch_dists)
            dists_np = all_dists.detach().cpu().numpy()
            new_R = get_radius(dists_np, self.nu)
            self.R = new_R
            self.log("R", self.R)
        # Clear stored distances for next epoch.
        self._train_epoch_dists = []

    def test_step(self, batch, batch_idx):
        inputs, labels, indices = batch
        outputs = self.forward(inputs)
        # Compute squared distances.
        dist = torch.sum((outputs - self.c) ** 2, dim=1)
        # For soft-boundary, adjust scores by subtracting R².
        if self.objective == "soft-boundary":
            scores = dist - self.R**2
        else:
            scores = dist

        output = {"indices": indices, "labels": labels, "scores": scores}
        self._test_outputs.append(output)
        return output

    def on_test_epoch_end(self):
        # Aggregate scores and labels from stored test outputs.
        if not self._test_outputs:
            return
        all_scores = torch.cat([x["scores"] for x in self._test_outputs]).cpu().numpy()
        all_labels = torch.cat([x["labels"] for x in self._test_outputs]).cpu().numpy()
        avg_score = all_scores.mean()
        self.log("avg_test_score", avg_score, prog_bar=True)
        auc = roc_auc_score(all_labels, all_scores)
        self.log("test_auc", auc, prog_bar=True)
        # Clear outputs for the next test epoch.
        self._test_outputs.clear()

    def configure_optimizers(self):
        # Create optimizer.
        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer_name == "amsgrad":
            optimizer = torch.optim.Adam(
                self.net.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                amsgrad=True,
            )
        else:
            raise ValueError("Unsupported optimizer")
        # Create a StepLR scheduler (alternatively, use MultiStepLR if milestones are provided).
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.lr_milestones, gamma=self.scheduler_gamma
        )
        return [optimizer], [scheduler]

    def init_center(self, train_loader):
        """
        Initialize the hypersphere center c as the mean output over the training set.
        """
        n_samples = 0
        c = None
        self.net.eval()
        device = self.device  # or self.trainer.strategy.root_device in Lightning v2
        with torch.no_grad():
            for batch in train_loader:
                inputs, _, _ = batch
                inputs = inputs.to(device)
                outputs = self.forward(inputs)
                if c is None:
                    c = torch.sum(outputs, dim=0)
                else:
                    c += torch.sum(outputs, dim=0)
                n_samples += outputs.shape[0]
        c /= n_samples
        # Adjust any values close to zero.
        eps = 0.1
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        self.c = c
        self.net.train()

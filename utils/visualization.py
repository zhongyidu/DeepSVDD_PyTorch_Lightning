import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid


def plot_images_grid(
    x: torch.Tensor,
    export_img: str,
    title: str = "",
    nrow=8,
    padding=2,
    normalize=False,
    pad_value=0,
):
    grid = make_grid(
        x, nrow=nrow, padding=padding, normalize=normalize, pad_value=pad_value
    )
    npgrid = grid.cpu().numpy()
    plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation="nearest")
    plt.title(title)
    plt.axis("off")
    plt.savefig(export_img, bbox_inches="tight")
    plt.close()

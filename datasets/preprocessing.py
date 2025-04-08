import torch
import numpy as np


def get_target_label_idx(labels, targets):
    return np.argwhere(np.isin(labels, targets)).flatten().tolist()


def global_contrast_normalization(x: torch.tensor, scale="l2"):
    assert scale in ("l1", "l2")
    n_features = int(np.prod(x.shape))
    mean = torch.mean(x)
    x = x - mean
    if scale == "l1":
        x_scale = torch.mean(torch.abs(x))
    else:
        x_scale = torch.sqrt(torch.sum(x**2)) / n_features
    x = x / x_scale
    return x

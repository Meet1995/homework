from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision import transforms


class CONFIG:
    batch_size = 390
    num_epochs = 12
    initial_weight_decay = 1e-5
    batches_per_epoch = int(50000 / batch_size)
    if 50000 % batch_size != 0:
        batches_per_epoch = int(50000 / batch_size) + 1

    lrs_kwargs = {
        "max_lr": 0.009,
        "ini_lr_div": 40,
        "final_lr_div": 500,
        "n_epochs": num_epochs,
        "batches_per_epoch": batches_per_epoch,
        "cycle_perct": 0.96,
    }
    initial_learning_rate = lrs_kwargs["max_lr"] / lrs_kwargs["ini_lr_div"]

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(
        model.parameters(),
        lr=CONFIG.initial_learning_rate,
        weight_decay=CONFIG.initial_weight_decay,
    )

    transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ]
    )

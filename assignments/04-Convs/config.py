from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision import transforms


class CONFIG:
    batch_size = 256
    num_epochs = 4

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(model.parameters(), lr=0.007)

    transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
        ]
    )

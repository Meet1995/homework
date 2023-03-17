import torch
from typing import Tuple
from config import CONFIG
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


class BaseModel(torch.nn.Module):
    """_summary_

    Args:
        torch (_type_): _description_
    """

    def __init__(self, num_channels, num_classes):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(num_channels, 16, 3, stride=2, padding=0),
            torch.nn.MaxPool2d(kernel_size=5, stride=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.Flatten(),
            torch.nn.LazyLinear(num_classes),
        )
        self.layers.apply(self._init_weights)

    def _init_weights(self, layer):
        if isinstance(layer, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        return self.layers(x)


class Model(torch.nn.Module):
    """_summary_

    Args:
        torch (_type_): _description_
    """

    def __init__(self, num_channels, num_classes):
        super().__init__()
        self.base_model = BaseModel(num_channels, num_classes)
        self.__pretrain()

    def __pretrain(self):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = CONFIG.optimizer_factory(self.base_model)
        train_loader, test_loader = self.__get_cifar10_data()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model.to(device)
        for epoch in range(2):
            self.base_model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                y_hat = self.base_model(x)
                loss = criterion(y_hat, y)
                loss.backward()
                optimizer.step()

    def __get_cifar10_data(self) -> Tuple[DataLoader, DataLoader]:
        train_data = CIFAR10(
            root="data/cifar10", train=True, download=False, transform=CONFIG.transforms
        )
        train_loader = DataLoader(
            train_data, batch_size=CONFIG.batch_size, shuffle=True
        )
        test_data = CIFAR10(
            root="data/cifar10",
            train=False,
            download=False,
            transform=CONFIG.transforms,
        )
        test_loader = DataLoader(test_data, batch_size=CONFIG.batch_size, shuffle=True)
        return train_loader, test_loader

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        return self.base_model(x)

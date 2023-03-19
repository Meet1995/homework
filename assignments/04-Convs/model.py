import torch


class Model(torch.nn.Module):
    """_summary_

    Args:
        torch (_type_): _description_
    """

    def __init__(self, num_channels, num_classes):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(num_channels, 16, 3, stride=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.BatchNorm2d(16),
            # torch.nn.Conv2d(32, 16, 3, stride=2, padding=0),
            # torch.nn.ReLU(),
            # torch.nn.BatchNorm2d(16),
            torch.nn.Flatten(),
            torch.nn.LazyLinear(128),
            torch.nn.Linear(128, num_classes),
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

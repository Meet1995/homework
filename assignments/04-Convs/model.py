import torch


class Model(torch.nn.Module):
    """_summary_

    Args:
        torch (_type_): _description_
    """

    def __init__(self, num_channels, num_classes):
        super().__init__()
        n_chn1 = 64
        n_chn2 = 128
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(num_channels, n_chn1, 3, stride=2, bias=False),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(n_chn1),
            torch.nn.Conv2d(n_chn1, n_chn2, 3, stride=2, bias=False),
            torch.nn.MaxPool2d(kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(n_chn2),
            torch.nn.Conv2d(n_chn2, 10, 5, stride=1),
            torch.nn.Flatten(),
        )
        self.layers.apply(self._init_weights)

    def _init_weights(self, layer):
        if isinstance(layer, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        return self.layers(x)

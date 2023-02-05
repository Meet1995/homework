import torch
from typing import Callable, Union


class MLP(torch.nn.Module):
    """
    Initialize the MLP.
    Args:
        input_size (int): The dimension D of the input data.
        hidden_size (Union[int, list]): The number of units K in every layer.
        For interger input, three layers with 64 units are created.
        num_classes (int): The number of classes C.
        hidden_count (int, optional): The number of units in the final hidden layer.
        activation (Callable, optional): The activation function to use in the
        hidden layer. Defaults to torch.nn.LeakyReLU.
        initializer (Callable, optional): The initializer to use for the weights.
        Defaults to torch.nn.init.xavier_normal_.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: Union[int, list],
        num_classes: int,
        hidden_count: int = 64,
        activation: Callable = torch.nn.LeakyReLU,
        initializer: Callable = torch.nn.init.xavier_normal_,
    ) -> None:
        super().__init__()

        self.actv = activation()
        self.initializer = initializer
        if isinstance(hidden_size, int):
            hidden_size = [64] * 3

        c1, c2, c3 = hidden_size

        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, c1, 3),
            activation(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(c1, c2, 3),
            activation(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(c2, c3, 3),
            activation(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(c3, hidden_count),
            activation(),
            torch.nn.Linear(hidden_count, num_classes),
        )
        self.layers.apply(self._init_weights)

    def _init_weights(self, layer):
        if isinstance(layer, (torch.nn.Linear, torch.nn.Conv2d)):
            self.initializer(layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        return self.layers(x.reshape(-1, 1, 28, 28))

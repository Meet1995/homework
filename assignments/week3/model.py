import torch
from typing import Callable


class MLP(torch.nn.Module):
    """
    Initialize the MLP.
    Args:
        input_size (int): The dimension D of the input data.
        hidden_size (int): The number of neurons H in the hidden layer.
        num_classes (int): The number of classes C.
        hidden_count (int, optional): The number of hidden layers. Defaults to 1.
        activation (Callable, optional): The activation function to use in the
        hidden layer. Defaults to torch.nn.ReLU.
        initializer (Callable, optional): The initializer to use for the weights.
        Defaults to initializer.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        super().__init__()

        self.actv = activation()

        if hidden_count >= 2:
            self.layers = torch.nn.ModuleList()
            for i in range(hidden_count):
                if i == 0:
                    layer = torch.nn.Linear(input_size, hidden_size)
                elif i == hidden_count - 1:
                    layer = torch.nn.Linear(hidden_size, num_classes)
                else:
                    layer = torch.nn.Linear(hidden_size, hidden_size)

                initializer(layer.weight)
                self.layers += [layer]

        elif hidden_count == 1:
            input_layer = torch.nn.Linear(input_size, hidden_size)
            initializer(input_layer.weight)

            output_layer = torch.nn.Linear(hidden_size, num_classes)
            initializer(output_layer.weight)

            self.layers = torch.nn.ModuleList([input_layer, output_layer])

        elif hidden_count == 0:
            layer = torch.nn.Linear(input_size, num_classes)
            initializer(layer.weight)

            self.layers = torch.nn.ModuleList([layer])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        for layer in self.layers:
            x = layer(x)
            x = self.actv(x)
        return x

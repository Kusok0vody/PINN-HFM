import math
import torch
import torch.nn as nn

from activations import Sine, Morlet, MexicanHat


_KAIMING_ACTIVATIONS = (nn.ReLU, nn.LeakyReLU, nn.ELU, nn.GELU, nn.SiLU)
_SIREN_ACTIVATIONS   = (Sine, Morlet, MexicanHat)


def _init_linear(layer: nn.Linear, activation, is_first: bool = False):
    """
    Selects weight initialisation based on the activation function.

    - Sine          → SIREN initialisation
    - ReLU-like     → Kaiming uniform
    - everything else → Xavier uniform
    """
    act_class = activation if isinstance(activation, type) else type(activation)

    if issubclass(act_class, _SIREN_ACTIVATIONS):
        act = activation() if isinstance(activation, type) else activation
        omega = act.omega.item() if isinstance(act.omega, torch.Tensor) else act.omega
        n = layer.in_features
        if is_first:
            bound = 1.0 / n
        else:
            bound = math.sqrt(6.0 / n) / omega
        nn.init.uniform_(layer.weight, -bound, bound)
        if layer.bias is not None:
            nn.init.uniform_(layer.bias, -bound, bound)

    elif issubclass(act_class, _KAIMING_ACTIVATIONS):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    else:
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


class MLP(nn.Module):
    def __init__(
        self,
        in_dim:     int,
        out_dim:    int,
        hidden_dim: int,
        num_layers: int,
        activation,
    ):
        super().__init__()

        layers = []
        dim    = in_dim

        for i in range(num_layers - 1):
            linear = nn.Linear(dim, hidden_dim)
            act    = activation() if isinstance(activation, type) else activation
            _init_linear(linear, activation, is_first=(i == 0))
            layers.append(linear)
            layers.append(act)
            dim = hidden_dim

        last = nn.Linear(dim, out_dim)
        _init_linear(last, activation, is_first=False)
        layers.append(last)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiMLP(nn.Module):
    def __init__(
        self,
        in_dim:     int,
        hidden_dim: int,
        num_layers: int,
        activation,
        K:          int,
    ):
        super().__init__()

        self.mlps = nn.ModuleList([
            MLP(in_dim, 1, hidden_dim, num_layers, activation)
            for _ in range(K)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([mlp(x) for mlp in self.mlps], dim=0).sum(dim=0)
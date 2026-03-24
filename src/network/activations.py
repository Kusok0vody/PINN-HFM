import torch
import torch.nn as nn


class Sine(nn.Module):
    def __init__(self, omega: float = 30.0, trainable: bool = False):
        super().__init__()
        if trainable:
            self.omega = nn.Parameter(torch.tensor(omega))
        else:
            self.register_buffer("omega", torch.tensor(omega))

    def forward(self, x):
        return torch.sin(self.omega * x)


class MexicanHat(nn.Module):
    def forward(self, x):
        return (1 - x**2) * torch.exp(-0.5 * x**2)


class Morlet(nn.Module):
    def __init__(self, omega: float = 5.0, trainable: bool = False):
        super().__init__()
        if trainable:
            self.omega = nn.Parameter(torch.tensor(omega))
        else:
            self.register_buffer("omega", torch.tensor(omega))

    def forward(self, x):
        return torch.cos(self.omega * x) * torch.exp(-0.5 * x**2)
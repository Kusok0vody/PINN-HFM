import torch
import torch.nn as nn


class ActivationFactory:
    """Callable that creates a new activation instance each time."""
    def __init__(self, cls, **kwargs):
        self.cls    = cls
        self.kwargs = kwargs

    def __call__(self):
        return self.cls(**self.kwargs)
    

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


ACTIVATION_REGISTRY = {
    "Tanh":             nn.Tanh,
    "ReLU":             nn.ReLU,
    "GELU":             nn.GELU,
    "SiLU":             nn.SiLU,
    "Sine":             Sine,
    "Morlet":           Morlet,
}


def serialize_activation(act) -> dict | str:
    """
    Serializes activation to a JSON-compatible dict.
    Accepts ActivationFactory, a bare class, or an instance.
    """
    if isinstance(act, ActivationFactory):
        return {"cls": act.cls.__name__, "kwargs": act.kwargs, "_factory": True}
    if isinstance(act, type):
        return {"cls": act.__name__, "kwargs": {}, "_factory": False}
    # instance (unlikely in config, but handle gracefully)
    return {"cls": type(act).__name__, "kwargs": {}, "_factory": False}


def deserialize_activation(data: dict | str):
    """
    Restores ActivationFactory or bare class from serialized dict.
    Returns ActivationFactory if _factory=True, else bare class.
    """
    cls = ACTIVATION_REGISTRY[data["cls"]]
    if data.get("_factory", False):
        return ActivationFactory(cls, **data["kwargs"])
    return cls
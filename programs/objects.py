import numpy as np 
import torch
import torch.nn as nn


torch.manual_seed(1234)


class Sin(nn.Module):
    """
    sin activation function for Neural Network
    """
    def __init__(self, f=1.0):
        super().__init__()
        self.name = 'Sin'
        self.f = f

    def forward(self, inp:torch.Tensor) -> torch.Tensor:
        return torch.sin(self.f * inp)


class Cos(nn.Module):
    """
    cos activation function for Neural Network
    """
    def __init__(self, f=1.0):
        super().__init__()
        self.name = 'Cos'
        self.f = f

    def forward(self, inp:torch.Tensor) -> torch.Tensor:
        return torch.cos(self.f * inp)
        

class Morlet(nn.Module):
    def __init__(self, initial_freq=7/4):
        super().__init__()
        self.freq = nn.Parameter(torch.tensor(initial_freq))
        self.name = 'Morlet'
    
    def forward(self, inp):
        c1 = torch.nn.functional.softplus(self.freq)
        c2 = -1/2
        return torch.cos(c1*inp)*torch.exp(inp*inp*c2)


class MexicanHat(nn.Module):
    """
    (1-x^2)e^(-x^2/2)
    """
    def __init__(self):
        super().__init__()
        self.name = 'MexicanHat'
    
    def forward(self, inp):
        c1 = -1/2
        return (1-inp*inp)*torch.exp(inp*inp*c1)


class GaussianWawelet(nn.Module):
    """
    -x*e^(-x^2/2)
    """
    def __init__(self):
        super().__init__()
        self.name = 'GaussianWawelet'
    
    def forward(self, inp):
        c1 = -1/2
        return -1*inp*torch.exp(inp*inp*c1)


class Width():
    def __init__(self,
                 func : str,
                 w    : float,
                 w1   : float = 1,
                 w2   : float = 1,
                 w3   : float = 1,
                 w4   : float = 1
                ):
        self.name = func
        self.good_names = (
            "const",
            "elliptic",
        )
        
        self.w  = w
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4

        if not(func in self.good_names):
            raise NameError('Function not found')

    
    def __call__(self, x=None, y=None):
        if self.name == "const":
            return self.const(x)
        elif self.name == "elliptic":
            return self.elliptic(x, y)

    def const(self, x):
        return torch.ones_like(x)
        
    def elliptic(self, x, y):
        return 1 + self.w3 * ((x * self.w1 / 1.5)**2 + ((y - 1/2) * self.w2  / 0.5)**2)
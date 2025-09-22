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


class OutputHook(list):
    """ Hook to capture module outputs.
    """
    def __call__(self, module, input, output):
        self.append(output)


class Optimizer():
    def __init__(self, name, parameters):
        self.name = name

        self.weight_decay = 1e-5

        self.max_iter = 100000
        self.max_eval = 100000
        self.history_size = 200
        self.tolerance_grad = 1e-12
        self.tolerance_change = 0.5 * np.finfo(float).eps
        self.line_search_fn="strong_wolfe"
        self.parameters = parameters
        
    def __call__(self, ):
        if self.name=='Adam':
            return torch.optim.Adam(self.parameters(), self.weight_decay)
        if self.name=='NAdam':
            return torch.optim.NAdam(self.parameters(), self.weight_decay)
        if self.name=='LBFGS':
            return torch.optim.LBFGS(self.parameters(),
                                     # lr=0.1, 
                                     max_iter         = self.max_iter, 
                                     max_eval         = self.max_eval, 
                                     history_size     = self.history_size,
                                     tolerance_grad   = self.tolerance_grad, 
                                     tolerance_change = self.tolerance_change,
                                     line_search_fn   = self.line_search_fn)
        else:
            raise NameError('Function not found')


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
        self.good_names = ('w_const',    'wconst',    'const',
                           'w_sincos',   'wsincos',   'sincos',
                           'w_elliptic', 'welliptic', 'elliptic',
                           'w_wave',     'wwave',     'wave'
                          )
        self.const_name    = ('w_const',    'wconst',    'const')
        self.sincos_name   = ('w_sincos',   'wsincos',   'sincos')
        self.elliptic_name = ('w_elliptic', 'welliptic', 'elliptic')
        self.elliptic__name = ('w_elliptic_', 'welliptic_', 'elliptic_')
        self.wave_name     = ('w_wave',     'wwave',     'wave')
        self.sineband_name = ('w_sineband', 'wsineband', 'sineband')

        self.good_names = self.const_name + self.sincos_name + self.elliptic__name + self.elliptic_name + self.wave_name + self.sineband_name
        
        self.w  = w
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4

        if not(func in self.good_names):
            raise NameError('Function not found')

    
    def __call__(self, x=None, y=None):
        if self.name in self.const_name:
            return self.const(x)
        elif self.name in self.sincos_name:
            return self.sincos(x, y)
        elif self.name in self.elliptic_name:
            return self.elliptic(x, y)
        elif self.name in self.elliptic__name:
            return self.elliptic(x, y)
        elif self.name in self.wave_name:
            return self.wave(x, y)
        elif self.name in self.sineband_name:
            return self.sineband(x, y)


    def const_(self, x):       return self.w*torch.ones_like(x)

    def const(self, x):        return torch.ones_like(x)

    def sincos(self, x, y):    return self.w + self.w1 * torch.cos(torch.pi*x) * torch.sin(torch.pi*y)
    
    def elliptic_(self, x, y): return self.w + self.w1 * ((x / self.w2)**2 + ((y - self.w4) / self.w3)**2)
        
    def elliptic(self, x, y):  return 1 + self.w3/self.w * ((x * self.w1 / 1.5)**2 + ((y - 1/2) * self.w2  / 0.5)**2)

    def wave(self, x, y):      return self.w + self.w1 * torch.sin(torch.pi*(self.w2*x-self.w3*y))

    def sineband(self, x, y):  return self.w + self.w1 * (1 + torch.cos(torch.pi*self.w2*y)) / 2 * (torch.sigmoid(200*(x-self.w3)) - torch.sigmoid(200*(x-self.w4)))
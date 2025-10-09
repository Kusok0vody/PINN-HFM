import torch


def derivative(dx, x, order=1)->torch.Tensor:
    """
    Calculates the derivative of a given Tensor.

    Parameters
    ----------
    dx : Tensor
        The function which must be differentiated.
    x : Tensor
        The variable with respect to which the derivative is calculated.
    order : int
        Order of derivative.

    Returns
    -------
    dx : Tensor
        Derivative of function.
    """
    for _ in range(order):
        dx = torch.autograd.grad(outputs=dx, inputs=x, grad_outputs=torch.ones_like(dx), create_graph=True, retain_graph=True)[0]
    return dx


def psi(y, chi, y_max)->torch.Tensor:
    return torch.where((y - 1 / 2).abs().round(decimals=5) <= chi / 2 / y_max, 1., 0.)


def viscosity(c:torch.Tensor, beta=-2.5)->torch.Tensor:
    """
    Calculates the viscosity value depending on the concentration using the Nolte relation:\n
    mu = mu0 * (1 - c / cmax)^beta
    """
    return  (1 - c) ** (beta)


def rho(c:torch.Tensor, G:float, r:float)->torch.Tensor:
    return c


def norm(f:torch.Tensor)->torch.Tensor:
    return (f - f.min()) / (f.max() - f.min())

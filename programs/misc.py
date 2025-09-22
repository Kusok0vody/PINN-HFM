from numpy import sin, cos, sinh, cosh, power, pi, where

import torch

def Poisson_analytical(mu, Q0, w, x, y, chi, H, L, K):
    """
    Returns an analytical solution to the Poisson equation at zero concentration.\n
    The accuracy of the solution depends on the number of terms in the sum.

    Parameters
    ----------
    mu : float
        Viscosity of medium
    Q0 : float
        Flow.
    w : float
        Gap width.
    x : array
        nd-array of x-axis values.
    y : array
        nd-array of y-axis values.
    chi : float
        Perforation interval length.
    H : float
        Height of medium.
    L : float
        Length of medium.
    K : int
        Number of terms in the sum.

    Returns
    -------
    p : array
        Array of solutions to the Poisson equation at given points.
    """
    pp = x / H
    for k in range(1,K):
        pp += 2 / (pi * k * chi) * power(-1, k%2) * sin(pi * k * chi / H) * cos( 2 * pi * k * y / H) * sinh( 2 * pi * k * (x - L / 2) / H) / (2 * pi * k / H * cosh(2 * pi * k * L / 2 / H))

    return 12 * mu * Q0 / power(w,3) * pp


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

# def psi(y, chi, y_max):
#     """
#     Returns a step function with a value of 1 within a given interval for lists and NumPy arrays.

#     Parameters
#     ----------
#     y : list of array
#         Array of coordinates.
#     chi : float
#         The length of the interval in which the function value is equal to 1.
#     y_max : float
#         Length of the whole interval.
#     """
#     return torch.where((y - y_max / 2).abs().round(decimals=5) <= chi / 2, 1., 0.)
def psi(y, chi, y_max):
    return torch.where((y - 1 / 2).abs().round(decimals=5) <= chi / 2 / y_max, 1., 0.)


def heaviside(x):
    """
    Returns Heaviside function with a value of 1 where x>=0 for lists and NumPy arrays.

    Parameters
    ----------
    x : list of array
        Array of coordinates.
    """
    return where(x>=0, 1, 0)


def th(x, k=1, c=0, b=0):
    """
    Returns hyperbolic tangent which is given by a formula\n
    tanh(k(x-c)) + b\n
    for Tensors.

    Parameters
    ----------

    x : Tensor
        Tensor of coordinates.
    k : float
        Multiplied by the coordinate value.
    c : float
        Coordinate shift.
    b : float
        Changes the value of a function.
    """
    return torch.tanh(k * (x - c)).round(decimals=5) + b


def psi_th(x, c1=0, c2=1):
    """
    Returns a smoothed step-like function with a value of 1 within the given boundaries for Tensor.
    Parameters
    ----------
    y : Tensor
        Tensor of coordinates.
    c1 : float
        Left boundary.
    c2 : float
        Right boundary.
    """
    return th(x=x, k=250, c=c1, b=1) * th(x=x, k=-250, c=c2, b=1) / 4


def heaviside_th(x):
    """
    Returns smoothed Heaviside function with a value of 1 where x>=0 for Tensor.

    Parameters
    ----------
    x : list of array
        Array of coordinates.
    """
    return 1 / (1 + 1 / torch.exp(250*x))


def viscosity(mu0, c, cmax, beta=-2.5):
    """
    Calculates the viscosity value depending on the concentration using the Nolte relation:\n
    mu = mu0 * (1 - c / cmax)^beta

    Parameters
    ----------
    mu0 : float
        Viscocity of the medium.
    c : float, array or Tensor
        Concentration value or field.
    cmax : float
        Maximum concentration.
    beta : float
        Degree for formula.
    """
    return mu0 * (1 - c / cmax) ** (beta)


def norm(f:torch.Tensor):
    return (f - f.min()) / (f.max() - f.min())

def extremum(x):
    a = x.max()
    b = x.min()
    if a.abs() >= b.abs():
        return a
    else:
        return b


def compare_viscosity(c1, c2, cmax, beta=-2.5):
    return viscosity(1,c1,cmax,beta) / viscosity(1,c2,cmax,beta)


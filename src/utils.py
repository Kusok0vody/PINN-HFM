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
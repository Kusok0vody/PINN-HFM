import torch


def derivative(dx: torch.Tensor, x: torch.Tensor, order: int = 1) -> torch.Tensor:
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
        dx = torch.autograd.grad(
             outputs=dx,
             inputs=x,
             grad_outputs=torch.ones_like(dx),
             create_graph=True,
             retain_graph=True
        )[0]
    return dx

def derivative_batched(f: torch.Tensor, x: torch.Tensor, order: int = 1) -> torch.Tensor:
    """
    Computes d^n f / dx^n for f of shape (N, M) and x of shape (N, 1).
    Returns (N, M).
    """
    for _ in range(order):
        grads = []
        for j in range(f.shape[1]):
            g = torch.autograd.grad(
                outputs=f[:, j:j+1],
                inputs=x,
                grad_outputs=torch.ones_like(f[:, j:j+1]),
                create_graph=True,
                retain_graph=True,
            )[0]
            grads.append(g)
        f = torch.cat(grads, dim=1)
    return f

def smooth_clamp(x: torch.Tensor, lo: float = 0.0, hi: float = 1.0, eps: float = 1e-3) -> torch.Tensor:
    """
    Smooth clamp to [lo, hi] via twice-applied softplus.
    Linear in the interior, smooth at the boundaries.
    """
    X = (x + torch.sqrt(x**2 + eps**2)) / 2
    return hi - (hi - lo) * ((hi - X + torch.sqrt((hi - X)**2 + eps**2)) / 2) / hi

COORD_ORDER = {
    (True,  2): ["t", "y", "x"],
    (True,  1): ["t", "x"],
    (False, 2): ["y", "x"],
    (False, 1): ["x"],
}

def unpack_coords(coords: torch.Tensor, has_time: bool, dim: int) -> dict:
    """
    Unpacks a coordinate tensor into a named dict.

    Args:
        coords:   (N, n_coords)
        has_time: whether time coordinate is present
        dim:      spatial dimensionality

    Returns:
        dict, e.g. {"t": (N,1), "y": (N,1), "x": (N,1)}
    """
    order = COORD_ORDER[(has_time, dim)]
    return {
        name: coords[:, i:i+1]
        for i, name in enumerate(order)
    }
    
def unpack_coords_grad(coords: torch.Tensor, has_time: bool, dim: int) -> dict:
    order = COORD_ORDER[(has_time, dim)]
    return {
        name: coords[:, i:i+1].requires_grad_(True)
        for i, name in enumerate(order)
    }
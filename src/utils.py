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

def _derivative_loop(f: torch.Tensor, x: torch.Tensor, order: int) -> torch.Tensor:
    """One autograd call per parameter setting. Kept as the reference path."""
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


def derivative_paired(f: torch.Tensor, x: torch.Tensor, order: int = 1) -> torch.Tensor:
    """
    d^n f / dx^n where f and x are both (N, M) and f[n, m] depends only on
    x[n, m]. Returns (N, M).

    With one leaf per (point, parameter) pair the whole Jacobian is diagonal, so
    a single backward on f.sum() collects every component — no loop over
    settings. This is the shortcut that is wrong when x is (N, 1) shared across
    settings, because there the sum collapses the parameter axis, and correct
    once it is not.
    """
    for _ in range(order):
        f = torch.autograd.grad(
            outputs=f.sum(), inputs=x,
            create_graph=True, retain_graph=True,
        )[0]
    return f


def unpack_coords_paired(coords: torch.Tensor, has_time: bool, dim: int, n_mu: int,
                         requires_grad: bool = True) -> tuple[dict, torch.Tensor]:
    """
    Unpack coordinates and replicate them across parameter settings.

    Args:
        coords: (N, n_coords)
        n_mu:   number of parameter settings M

    Returns:
        unpacked: {name: (N, M)} leaves
        X:        (N, M, n_coords) for Net.forward_paired
    """
    unpacked = {}
    for i, name in enumerate(COORD_ORDER[(has_time, dim)]):
        col = coords[:, i:i+1].detach().expand(-1, n_mu).contiguous()
        if requires_grad:
            col = col.requires_grad_(True)
        unpacked[name] = col
    return unpacked, torch.stack(list(unpacked.values()), dim=-1)


def derivative_batched(f: torch.Tensor, x: torch.Tensor, order: int = 1) -> torch.Tensor:
    """
    Computes d^n f / dx^n for f of shape (N, M) and x of shape (N, 1).
    Returns (N, M).

    When x has the same shape as f the coordinates carry one leaf per
    (point, parameter) pair and a single backward suffices. Every physics module
    reaches that path without changing a line, because the dispatch is on shape
    rather than on a flag threaded through the call sites.

    Otherwise the coordinates are shared across settings and the components have
    to be separated one at a time. A batched-seed variant of that loop was tried
    and removed: measured on the real workload (second derivative plus backward
    to the weights) it ran at 0.8-1.1x the loop's speed and used the same
    memory, so it bought nothing for an extra code path.
    """
    if x.shape == f.shape and x.dim() == 2 and x.shape[1] > 1:
        return derivative_paired(f, x, order)
    return _derivative_loop(f, x, order)

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

def unpack_coords(coords: torch.Tensor, has_time: bool, dim: int, 
                  requires_grad: bool = False) -> tuple[dict, torch.Tensor]:
    """
    Unpacks a coordinate tensor into a named dict and reconstructs the tensor.

    Args:
        coords:       (N, n_coords)
        has_time:     whether time coordinate is present
        dim:          spatial dimensionality
        requires_grad: whether to set requires_grad=True on each column

    Returns:
        unpacked: dict {name: (N, 1)} with requires_grad if requested
        coords:   (N, n_coords) reconstructed from unpacked columns
    """
    unpacked = {}
    for i, name in enumerate(COORD_ORDER[(has_time, dim)]):
        col = coords[:, i:i+1].detach()
        if requires_grad:
            col = col.requires_grad_(True)
        unpacked[name] = col
    coords_out = torch.cat(list(unpacked.values()), dim=1)
    return unpacked, coords_out
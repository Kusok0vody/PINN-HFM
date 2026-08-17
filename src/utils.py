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

VECTORISE_DERIVATIVES = True


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


def _derivative_vmap(f: torch.Tensor, x: torch.Tensor, order: int) -> torch.Tensor:
    """
    The same derivatives, one vmapped autograd call per order.

    The obvious shortcut is wrong: grad(f.sum(), x) collapses the parameter axis
    and returns sum_m df[:, m]/dx, not the individual components. Recovering
    them by slicing is what the loop does, one call at a time.

    Batched grad_outputs instead seeds M vector-Jacobian products at once, row m
    of the seed asking for df[:, m]/dx. The result arrives as (M, N, 1) and is
    transposed back.

    create_graph is kept so the output stays differentiable with respect to the
    network weights — verified against the loop to 2e-7 on the parameter
    gradients, which is the property that silently breaks if the graph is cut.
    """
    for _ in range(order):
        n, m = f.shape
        seeds = torch.eye(m, dtype=f.dtype, device=f.device)
        seeds = seeds.unsqueeze(1).expand(m, n, m)
        g = torch.autograd.grad(
            outputs=f, inputs=x, grad_outputs=seeds,
            is_grads_batched=True, create_graph=True, retain_graph=True,
        )[0]
        f = g.squeeze(-1).transpose(0, 1)
    return f


def derivative_batched(f: torch.Tensor, x: torch.Tensor, order: int = 1) -> torch.Tensor:
    """
    Computes d^n f / dx^n for f of shape (N, M) and x of shape (N, 1).
    Returns (N, M).

    Set utils.VECTORISE_DERIVATIVES = False to fall back to the explicit loop,
    which is slower by roughly a factor of three but depends on nothing beyond
    plain autograd.
    """
    if not VECTORISE_DERIVATIVES or f.shape[1] == 1:
        return _derivative_loop(f, x, order)
    return _derivative_vmap(f, x, order)

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
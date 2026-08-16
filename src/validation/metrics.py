import torch
import numpy as np


def _flatten(a) -> torch.Tensor:
    """Accept a numpy array or a torch tensor, return a flat float64 CPU tensor."""
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a)
    return a.detach().to(torch.float64).flatten().cpu()


def relative_l2(pred, ref, eps: float = 1e-12) -> float:
    """
    Relative L2 error  ||pred - ref|| / ||ref||.

    Accepts numpy arrays or torch tensors in any matching shape — both are
    flattened first, so a (N_t, N_x) grid and a flat (N,) vector behave the same.

    Args:
        pred: predicted field
        ref:  reference field, same shape as pred
        eps:  guard against a zero-norm reference

    Returns:
        error as a plain float
    """
    p = _flatten(pred)
    r = _flatten(ref)
    if p.shape != r.shape:
        raise ValueError(
            f"shape mismatch: pred {tuple(p.shape)} vs ref {tuple(r.shape)}"
        )
    return (torch.linalg.norm(p - r) / (torch.linalg.norm(r) + eps)).item()


def max_abs_error(pred, ref) -> float:
    """Worst-case pointwise error. Complements relative_l2, which averages."""
    p = _flatten(pred)
    r = _flatten(ref)
    if p.shape != r.shape:
        raise ValueError(
            f"shape mismatch: pred {tuple(p.shape)} vs ref {tuple(r.shape)}"
        )
    return (p - r).abs().max().item()


def residual_norms(residuals: dict) -> dict:
    """
    Flatten the nested dict returned by PINN.step() into {"group/name": mse}.

    Adds a "total" key with the unweighted sum, so runs stay comparable even
    when the adaptive loss weights differ between them.
    """
    flat  = {}
    total = 0.0

    for group, res_dict in residuals.items():
        for name, res in res_dict.items():
            val = (res ** 2).mean().item()
            flat[f"{group}/{name}"] = val
            total += val

    flat["total"] = total
    return flat

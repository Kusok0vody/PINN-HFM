"""
The Lotka-Volterra sweep, in one place.

Training and evaluation have to agree on eleven ranges and on the time horizon.
Written out twice they would agree until one of them was edited, and the
disagreement would not announce itself: an evaluation drawing settings from
ranges the run never trained on reads as a network that generalises badly,
which is indistinguishable from a network that does.
"""
import torch

T_END = 30.0

# Eleven swept axes: eight rates and the three initial values.
#
# The initial values are parameters like any other here. They have to be, for
# the hard initial condition to mean anything: an ansatz built around a fixed
# u0 would hold that one Cauchy datum exactly and know nothing about any other,
# which is the opposite of a parametric solution.
LIMITS = {
    "r":     {"min": 0.5,  "max": 2.0,  "scale": "linear"},
    "K":     {"min": 0.8,  "max": 1.2,  "scale": "linear"},
    "alpha": {"min": 0.2,  "max": 1.5,  "scale": "linear"},
    "e1":    {"min": 0.2,  "max": 0.8,  "scale": "linear"},
    "d1":    {"min": 0.1,  "max": 0.6,  "scale": "linear"},
    "beta":  {"min": 0.1,  "max": 1.2,  "scale": "linear"},
    "e2":    {"min": 0.2,  "max": 0.9,  "scale": "linear"},
    "d2":    {"min": 0.05, "max": 0.4,  "scale": "linear"},
    "G0":    {"min": 0.2,  "max": 1.0,  "scale": "linear"},
    "H0":    {"min": 0.1,  "max": 0.6,  "scale": "linear"},
    "P0":    {"min": 0.05, "max": 0.4,  "scale": "linear"},
}


def limits(n: int) -> dict:
    """The sweep definition at a given batch size."""
    return {k: dict(v, N=n) for k, v in LIMITS.items()}


def midpoint() -> dict:
    """One setting at the centre of every range, to construct a physics with."""
    return {k: 0.5 * (v["min"] + v["max"]) for k, v in LIMITS.items()}


def bounds() -> dict:
    """
    The two faces of the time interval.

    Neither carries a loss term: the initial condition is in the ansatz and the
    far end is free, so no boundary points are drawn at all. They are here
    because the geometry is defined by its faces, not because anything is
    imposed on them.
    """
    return {
        "ic":    {"p": [0.0, 1.0], "x": lambda p: torch.zeros_like(p),        "N": 0},
        "right": {"p": [0.0, 1.0], "x": lambda p: T_END * torch.ones_like(p), "N": 0},
    }

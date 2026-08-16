import torch.nn.functional as F

from physics.phys import Physics
from utils import derivative_batched
from geometry.sampler import BoundaryBatch


class LotkaVolterra(Physics):
    """
    Three-species Lotka-Volterra system (grass G, herbivores H, predators P).

    Implemented as a time-free 1D problem (dim=1, has_time=False): the t axis
    is treated as the "x" coordinate, the Cauchy initial data is imposed on the
    left boundary (x=0), the right boundary (x=T) is free. Outputs are passed
    through softplus to keep G, H, P strictly positive.

    Variables:
        G — grass biomass
        H — herbivore population
        P — predator population

    PDE:
        dG/dt = r * G * (1 - G/K) - alpha * G * H
        dH/dt = e1 * alpha * G * H - d1 * H - beta * H * P
        dP/dt = e2 * beta * H * P - d2 * P

    Boundary conditions:
        dirichlet on x=0 (initial values of G, H, P)

    Initial condition:
        imposed via the x=0 boundary

    Parameters (each is a tensor of shape (M,) for M parameter settings):
        r     — intrinsic growth rate of grass
        K     — carrying capacity of grass
        alpha — grazing rate (consumption of G by H)
        e1    — conversion efficiency of grass into herbivore biomass
        d1    — death rate of herbivores
        beta  — predation rate (consumption of H by P)
        e2    — conversion efficiency of herbivores into predator biomass
        d2    — death rate of predators
    """

    def __init__(self, device="cpu", dim=1, has_time=False):
        super().__init__(device, dim, has_time)

    def setParameters(
        self,
        params: list[dict],
        boundaries: dict,
        initial: dict = None,
        limits: dict = {},
    ):
        required = {"r", "K", "alpha", "e1", "d1", "beta", "e2", "d2"}
        missing  = required - set(params[0].keys())
        if missing:
            raise ValueError(f"Missing parameters: {missing}")

        self.param_order = list(params[0].keys())
        self.par         = self.make_param_batch(params)
        self.limits      = limits

        self.boundaries = {
            name: {
                var: cond
                for var, cond in bound.get("bc", {}).items()
                if isinstance(cond, dict) and "type" in cond
            }
            for name, bound in boundaries.items()
            if bound.get("bc")
        }

        self.boundary_meta = {
            name: {
                "periodic": bound.get("periodic", False),
                "with":     bound.get("with", None),
            }
            for name, bound in boundaries.items()
        }

        self.initial = initial or {}

        self.transforms = {
            "G": lambda u: F.softplus(u),
            "H": lambda u: F.softplus(u),
            "P": lambda u: F.softplus(u),
        }

    def residualPDE(self, pred: dict, unpacked: dict) -> dict:
        t = unpacked["x"]

        G = pred["G"]
        H = pred["H"]
        P = pred["P"]

        r     = self.par["r"].unsqueeze(0)
        K     = self.par["K"].unsqueeze(0)
        alpha = self.par["alpha"].unsqueeze(0)
        e1    = self.par["e1"].unsqueeze(0)
        d1    = self.par["d1"].unsqueeze(0)
        beta  = self.par["beta"].unsqueeze(0)
        e2    = self.par["e2"].unsqueeze(0)
        d2    = self.par["d2"].unsqueeze(0)

        dG = derivative_batched(G, t)
        dH = derivative_batched(H, t)
        dP = derivative_batched(P, t)

        return {
            "G_eq": dG - (r * G * (1 - G / K) - alpha * G * H),
            "H_eq": dH - (e1 * alpha * G * H - d1 * H - beta * H * P),
            "P_eq": dP - (e2 * beta * H * P - d2 * P),
        }

    def residualBC(self, pred: dict, coords_bc: dict, batch: BoundaryBatch) -> dict:
        if batch.name not in self.boundaries:
            return {}

        x = coords_bc["x"]

        residuals = {}
        for var_name, cond in self.boundaries[batch.name].items():
            if cond["type"] == "dirichlet":
                val = cond["value"](x)
                residuals[f"{batch.name}_{var_name}"] = pred[var_name] - val

        return residuals

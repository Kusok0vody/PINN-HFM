import math
import torch

from physics.phys import Physics
from utils import derivative_batched
from geometry.sampler import BoundaryBatch


class CDR1D(Physics):
    """
    1D convection-diffusion-reaction equation.

    Variables:
        u — transported scalar

    PDE:
        du/dt + beta * du/dx - nu * d2u/dx2 - rho * u * (1 - u) = 0

    Boundary conditions:
        u: periodic, u(t, 0) = u(t, 2*pi)

    Initial condition:
        u = (1 / (sigma * sqrt(2*pi))) * exp(-(x - pi)^2 / (2 * sigma^2)),  sigma = pi/2

    Parameters (each is a tensor of shape (M,) for M parameter settings):
        beta: convection speed
        nu:   diffusion coefficient
        rho:  reaction rate
    """

    def __init__(self, device="cpu", dim=1, has_time=True):
        super().__init__(device, dim, has_time)

    def setParameters(self, params: list[dict], boundaries: dict,
                      initial: dict = None, limits: dict = {}, hard_ic=False):
        required = {"beta", "nu", "rho"}
        missing  = required - params[0].keys()
        if missing:
            raise ValueError(f"Missing parameters: {missing}")

        self.param_order = list(params[0].keys())
        self.par = self.make_param_batch(params)
        self.limits = limits

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

        sigma      = math.pi / 4.0
        default_ic = lambda x: torch.exp(-(x - math.pi) ** 2 / (2.0 * sigma ** 2))
        init_fn    = (initial or {}).get("u", default_ic)

        def init_fn_prime(x: torch.Tensor) -> torch.Tensor:
            """
            d(init_fn)/dx by autograd rather than a closed form.

            The hard-IC ansatz for ux must be the derivative of whatever initial
            condition is actually in use; a hand-written formula silently stays
            the derivative of the default Gaussian when a custom one is passed,
            and residualPDE then builds u_xx from the wrong function. The graph
            is kept alive because residualPDE differentiates ux again.
            """
            with torch.enable_grad():
                xg = x if x.requires_grad else x.detach().requires_grad_(True)
                grad, = torch.autograd.grad(
                    init_fn(xg).sum(), xg, create_graph=True
                )
            return grad


        if hard_ic:
            self.initial       = {}
            self.output_ansatz = {
                "u": lambda u, coords: init_fn(coords["x"]) + (1.0 - torch.exp(-coords["t"])) * u,
                "ux": lambda ux, coords: init_fn_prime(coords["x"]) + (1.0 - torch.exp(-coords["t"])) * ux,
            }
        else:
            self.initial       = initial or {"u": init_fn}
            self.output_ansatz = {}
        self.transforms = {"u": lambda u: u}

    def residualPDE(self, pred: dict, unpacked: dict) -> dict:
        t = unpacked["t"]
        x = unpacked["x"]
        u = pred["u"]
        ux = pred["ux"]

        beta = self.par["beta"].unsqueeze(0)
        nu   = self.par["nu"].unsqueeze(0)
        rho  = self.par["rho"].unsqueeze(0)

        u_t  = derivative_batched(u, t)
        u_x  = derivative_batched(u, x)
        u_xx = derivative_batched(ux, x)

        return {
            "cdr": u_t + beta * u_x - nu * u_xx - rho * u * (1.0 - u),
            "corr": u_x - ux
        }

    def residualBC(self, pred: dict, coords_bc: dict, batch: BoundaryBatch) -> dict:
        if batch.name not in self.boundaries:
            return {}

        t = coords_bc["t"]
        x = coords_bc["x"]
        u = pred["u"]

        residuals = {}
        for var_name, cond in self.boundaries[batch.name].items():
            key   = f"{batch.name}_{var_name}"
            value = cond.get("value", None)
            val   = value(t, x) if value is not None else torch.zeros_like(u)

            if cond["type"] == "dirichlet":
                residuals[key] = u - val

        return residuals

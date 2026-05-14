import torch

from src.physics.phys import Physics
from src.utils import derivative_batched
from src.geometry.sampler import BoundaryBatch


class helmholtz2D_annulus(Physics):
    """
    2D Helmholtz equation on an annulus.

    Variables:
        u — scalar field

    PDE:
        d2u/dx2 + d2u/dy2 + k * u = 0

    Boundary conditions:
        dirichlet / neumann per variable

    Initial condition:
        none (stationary problem)

    Parameters (each is a tensor of shape (M,) for M parameter settings):
        k: reaction / wave parameter
    """

    def __init__(self, device="cpu", dim=2, has_time=False):
        super().__init__(device, dim, has_time)

    def setParameters(self, params: list[dict], boundaries: dict,
                      initial: dict = None, limits: dict = {}):
        required = {"k"}
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
        }
        self.boundary_meta = {
            name: {
                "periodic": bound.get("periodic", False),
                "with":     bound.get("with", None),
            }
            for name, bound in boundaries.items()
        }

        self.transforms = {"u": lambda u: u}

    def residualPDE(self, pred: dict, unpacked: dict) -> dict:
        x = unpacked["x"]
        y = unpacked["y"]
        u = pred["u"]

        k = self.par["k"].unsqueeze(0)

        u_x = derivative_batched(u, x)
        u_y = derivative_batched(u, y)
        u_xx = derivative_batched(u_x, x)
        u_yy = derivative_batched(u_y, y)

        return {"helmholtz": u_xx + u_yy + k * u}

    def residualBC(self, pred: dict, coords_bc: dict, batch: BoundaryBatch) -> dict:
        if batch.name not in self.boundaries:
            return {}

        x  = coords_bc["x"]
        y  = coords_bc["y"]
        nx = batch.nx
        ny = batch.ny
        u  = pred["u"]

        residuals = {}
        for var_name, cond in self.boundaries[batch.name].items():
            if var_name != "u":
                continue
            key   = f"{batch.name}_{var_name}"
            btype = cond["type"]
            value = cond.get("value", None)
            val   = value(x, y) if value is not None else None

            if btype == "dirichlet":
                residuals[key] = u - val
            elif btype == "neumann":
                u_n    = derivative_batched(u, x) * nx + derivative_batched(u, y) * ny
                target = val if val is not None else torch.zeros_like(u_n)
                residuals[key] = u_n - target

        return residuals

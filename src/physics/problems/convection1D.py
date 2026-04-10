import torch
from src.physics.phys import Physics
from src.physics.parameters import *
from src.utils import derivative_batched, unpack_coords
from src.geometry.sampler import BoundaryBatch

class convection1D(Physics):
    """
    1D linear convection equation:
        du/dt + beta * du/dx = 0

    Variables:
        u: transported scalar

    Parameters:
        beta: convection speed

    Boundary conditions:
        u: periodic

    Initial condition:
        u = u0(x), for example, 1 + sin(x)
    """

    def __init__(self, device="cpu", dim=1, has_time=True):
        super().__init__(device, dim, has_time)

    def setParameters(self, params: list[dict], boundaries: dict, initial: dict = None):
        required = {"beta"}
        missing = required - params[0].keys()
        if missing:
            raise ValueError(f"Missing parameters: {missing}")

        self.param_order = list(params[0].keys())
        self.par = self.make_param_batch(params)

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

        self.initial = initial or {
            "u": lambda x: torch.zeros_like(x)
        }

        self.transforms = {
            "u": lambda u: u
        }

    def residualPDE(self, pred: dict, unpacked: dict) -> dict:
        t = unpacked["t"]
        x = unpacked["x"]

        u = pred["u"]  # (N, M)

        beta = self.par["beta"].unsqueeze(0)  # (1, M)

        u_t = derivative_batched(u, t)
        u_x = derivative_batched(u, x)

        return {
            "convection": u_t + beta * u_x
        }

    def residualBC(self, pred: dict, coords_bc: torch.Tensor, batch: BoundaryBatch) -> dict:
        if batch.name not in self.boundaries:
            return {}

        unpacked, _ = unpack_coords(coords_bc, self.has_time, self.dim)

        t = unpacked["t"]
        x = unpacked["x"]
        nx = batch.nx

        u = pred["u"]

        residuals = {}

        for var_name, cond in self.boundaries[batch.name].items():
            key = f"{batch.name}_{var_name}"
            btype = cond["type"]
            value = cond.get("value", None)

            val = value(t, x) if value is not None else None

            if var_name == "u":
                if btype == "dirichlet":
                    residuals[key] = u - val

                elif btype == "neumann":
                    u_x = derivative_batched(u, x)
                    u_n = u_x * nx

                    target = val if val is not None else torch.zeros_like(u_n)
                    residuals[key] = u_n - target

        return residuals
    
    
import torch
from abc import ABC, abstractmethod

from geometry.sampler import BoundaryBatch
from physics.parameters import ParamBatch
from utils import derivative_batched, unpack_coords


class Physics(ABC):
    """
    Physics module for PINN. Abstract class
    """

    def __init__(self, device="cpu", dim=2, has_time=True):
        self.device      = device
        self.dim         = dim
        self.has_time    = has_time
        self.transforms  = {}
        self.boundaries  = {}
        self.initial     = {}
        self.param_order = []
        self.par: ParamBatch = None
    
    def make_param_batch(self, par: list) -> ParamBatch:
        return ParamBatch.from_dict(
            {key: [m[key] for m in par] for key in self.param_order},
            order=self.param_order,
            device = self.device
        )
    
    @abstractmethod
    def setParameters(self, params: dict, boundaries: dict, initial: dict = None):
        pass

    @abstractmethod
    def residualPDE(self, pred: dict, coords: torch.Tensor) -> dict:
        pass
    
    def residualBC(self, pred: dict, coords_bc: torch.Tensor, batch: BoundaryBatch) -> dict:
        if batch.name not in self.boundaries:
            return {}

        unpacked, _ = unpack_coords(coords_bc, self.has_time, self.dim)

        t  = unpacked["x"]
        x  = unpacked["x"]
        y  = unpacked["y"]
        nx = batch.nx      # (N, 1)
        ny = batch.ny      # (N, 1)

        residuals = {}

        for var_name, cond in self.boundaries[batch.name].items():
            key = f"{batch.name}_{var_name}"

            if cond.get("splitted", False):
                vx, vy = cond["components"]
                u_n    = pred[vx] * nx + pred[vy] * ny
                residuals[key] = u_n - cond["value"](t, x, y)

            else:
                u = pred[var_name]

                if cond["type"] == "dirichlet":
                    residuals[key] = u - cond["value"](t, x, y)

                elif cond["type"] == "neumann":
                    u_x = derivative_batched(u, x)
                    u_y = derivative_batched(u, y)
                    u_n = u_x * nx + u_y * ny
                    residuals[key] = u_n - cond["value"](t, x, y)

        return residuals

    def residualIC(self, pred: dict, coords: torch.Tensor) -> dict:
        unpacked, _ = unpack_coords(coords, self.has_time, self.dim)
        x = unpacked["x"]
        y = unpacked.get("y")

        return {
            name: pred[name] - ic(x, y)
            for name, ic in self.initial.items()
        }

    def residualExtra(self, pred: dict, coords: torch.Tensor) -> dict:
        return {}

    def apply_transforms(self, pred: dict) -> dict:
        """
        Applies output transforms to network predictions.
        Transforms are defined in self.transforms as {name: callable}.
        Variables without a transform are passed through unchanged.
        """
        return {
            name: self.transforms[name](val) if name in self.transforms else val
            for name, val in pred.items()
        }
import torch
from abc import ABC, abstractmethod

from utils import derivative_batched, unpack_coords


class Physics(ABC):
    """
    Physics module for PINN. Abstract class
    """

    def __init__(self, device="cpu", dim=2, has_time=True):
        self.device     = device
        self.dim        = dim
        self.has_time   = has_time
        self.par        = {}
        self.transforms = {}
        self.boundaries = {}
        self.initial    = {}
    
    @abstractmethod
    def setParameters(self, params: dict, boundaries: dict, initial: dict = None):
        pass

    @abstractmethod
    def residualPDE(self, pred: dict, coords: torch.Tensor, M: int) -> dict:
        pass
    
    @abstractmethod
    def residualBC(self, pred: dict, sampled_boundaries: dict, M: int) -> dict:
        residuals = {}

        for bound_name, batch in sampled_boundaries.items():
            if bound_name not in self.boundaries:
                continue

            unpacked = unpack_coords(batch.coords, self.has_time, self.dim)
            t  = unpacked.get("t")
            x  = unpacked["x"].requires_grad_(True)
            y  = unpacked.get("y")
            if y is not None:
                y = y.requires_grad_(True)
            nx, ny = batch.nx, batch.ny

            for var_name, cond in self.boundaries[bound_name].items():
                key = f"{bound_name}_{var_name}"

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

    def residualIC(self, pred: dict, coords: torch.Tensor, M: int) -> dict:
        unpacked = unpack_coords(coords, self.has_time, self.dim)
        x = unpacked["x"]
        y = unpacked.get("y")

        return {
            name: pred[name] - ic(x, y)
            for name, ic in self.initial.items()
        }
    
    @abstractmethod
    def residualExtra(self, pred: dict, coords: torch.Tensor, M: int) -> dict:
        return {}

    @abstractmethod
    def loss(self, residuals: dict, weights: dict) -> torch.Tensor:
        pass
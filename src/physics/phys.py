import torch
from abc import ABC, abstractmethod

from geometry.sampler import BoundaryBatch
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

    def set_par(self) -> torch.Tensor:
        """
        Converts dictionary of parameter tensors to a single parameter tensor
        and updates self.par with individual parameter tensors.

        Returns:
            par tensor (M, n_params) for passing to net
        """
        order = getattr(self, 'param_order', list(self.par.keys()))
        
        par_tensor = torch.stack([self.par[key] for key in order], dim=1)
        
        for i, key in enumerate(order):
            self.par[key] = par_tensor[:, i]
        
        return par_tensor

    @abstractmethod
    def residualPDE(self, pred: dict, coords: torch.Tensor, M: int) -> dict:
        pass
    
    # @abstractmethod
    def residualBC(self, pred: dict, batch: BoundaryBatch, M: int) -> dict:
        if batch.name not in self.boundaries:
            return {}

        unpacked = unpack_coords(batch.coords, self.has_time, self.dim)
        t  = unpacked.get("t")
        x  = unpacked["x"].requires_grad_(True)
        y  = unpacked.get("y")
        if y is not None:
            y = y.requires_grad_(True)
        nx, ny = batch.nx, batch.ny

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

    def residualIC(self, pred: dict, coords: torch.Tensor, M: int) -> dict:
        unpacked = unpack_coords(coords, self.has_time, self.dim)
        x = unpacked["x"]
        y = unpacked.get("y")

        return {
            name: pred[name] - ic(x, y)
            for name, ic in self.initial.items()
        }
    
    # @abstractmethod
    def residualExtra(self, pred: dict, coords: torch.Tensor, M: int) -> dict:
        return {}

    # @abstractmethod
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
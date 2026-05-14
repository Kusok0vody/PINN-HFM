import torch
from abc import ABC, abstractmethod
from math import log

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
        self.limits      = {}
        
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
    
    def residualBC(self, pred: dict, coords_bc: dict, batch: BoundaryBatch) -> dict:
        if batch.name not in self.boundaries:
            return {}

        t  = coords_bc["t"]
        x  = coords_bc["x"]
        y  = coords_bc["y"]
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
        y = unpacked.get("y", None)

        def eval_ic(ic_func):
            if self.dim == 1:
                return ic_func(x)
            else:
                return ic_func(x, y)

        return {
            name: pred[name] - eval_ic(ic)
            for name, ic in self.initial.items()
        }
    
    def residualExtra(self, pred: dict, coords: torch.Tensor) -> dict:
        return {}

    def apply_boundary_constraints(self, res_bc: dict) -> dict:
        meta  = getattr(self, "boundary_meta", {})
        final = {}
        used  = set()

        for name, m in meta.items():
            if name in used or not m.get("periodic", False):
                continue

            pair = m["with"]
            for key, val in res_bc.items():
                bname, var = key.split("_", 1)
                if bname != name:
                    continue
                k2 = f"{pair}_{var}"
                if k2 in res_bc:
                    final[f"periodic_{name}_{pair}_{var}"] = val - res_bc[k2]

            used.add(name)
            used.add(pair)

        for key, val in res_bc.items():
            bname = key.split("_", 1)[0]
            if bname not in used:
                final[key] = val

        return final

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
        
    def _resample_parameters(self) -> None:
        """
        Resamples only parameters that have limits defined.
        All limit entries must share the same N (batch size).
        """
        if not hasattr(self, "limits") or not self.limits:
            return

        n = next(iter(self.limits.values()))["N"]
        new_params = [{} for _ in range(n)]

        for key, lim in self.limits.items():
            lo, hi = lim["min"], lim["max"]
            scale  = lim.get("scale", "log")

            if scale == "log":
                vals = torch.exp(
                    torch.rand(n) * (log(hi) - log(lo)) + log(lo)
                )
            else:
                vals = torch.rand(n) * (hi - lo) + lo

            for i, v in enumerate(vals.tolist()):
                new_params[i][key] = v

        for key in self.param_order:
            if key not in self.limits:
                fixed_val = self.par[key][0].item()
                for p in new_params:
                    p[key] = fixed_val

        self.par = self.make_param_batch(new_params)
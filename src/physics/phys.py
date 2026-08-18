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
        self.device        = device
        self.dim           = dim
        self.has_time      = has_time
        self.transforms    = {}
        self.boundaries    = {}
        self.initial       = {}
        self.param_order   = []
        self.par: ParamBatch = None
        self.limits        = {}
        self.output_ansatz = {}
        
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
    
    def residualExtra(self, pred: dict, coords: dict) -> dict:
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
        
    def apply_output_ansatz(self, pred: dict, coords: dict) -> dict:
        if not self.output_ansatz:
            return pred
        return {
            name: self.output_ansatz[name](val, coords) if name in self.output_ansatz else val
            for name, val in pred.items()
        }
        
    def draw_parameters(self, n: int = None, anchor_ends: bool = True) -> list[dict]:
        """
        Draw n parameter settings from the declared limits, without installing
        them. Parameters with no limits keep their current value.

        Split out of _resample_parameters so that a caller can draw a candidate
        pool, score it, and keep only the settings worth training on.
        """
        if not getattr(self, "limits", None):
            return []

        sizes = {lim["N"] for lim in self.limits.values()}
        if len(sizes) > 1:
            raise ValueError(
                f"all limit entries must share the same N, got {sorted(sizes)}"
            )
        n = sizes.pop() if n is None else n
        new_params = [{} for _ in range(n)]

        n_axes = len(self.limits)
        if anchor_ends and n < 2 * n_axes:
            print(f"Physics: {n} settings is fewer than the {2 * n_axes} needed "
                  f"to pin both ends of {n_axes} swept parameters, so the ends "
                  f"are left to the random draw and will never be hit exactly.")

        for axis_index, (key, lim) in enumerate(self.limits.items()):
            lo, hi = float(lim["min"]), float(lim["max"])
            # "linear" is the safe default: a log sweep needs a positive lower
            # bound, and min = 0 is common enough that defaulting to log turns
            # an omitted key into a bare "math domain error".
            scale = lim.get("scale", "linear")

            if scale == "log":
                if lo <= 0.0:
                    raise ValueError(
                        f"limits['{key}'] uses scale='log' but min={lo}; "
                        "a log sweep needs a strictly positive lower bound."
                    )
                vals = torch.exp(torch.rand(n) * (log(hi) - log(lo)) + log(lo))
            else:
                vals = torch.rand(n) * (hi - lo) + lo

            # Pin the ends of every axis into the batch.
            #
            # A uniform draw never returns min or max, so the boundary of the
            # declared sweep is the one place the network is always
            # extrapolating: an interior value has neighbours on both sides, an
            # endpoint has them on one. Measured on the Helmholtz annulus swept
            # over [1, 6], the amplitude ratio ran 0.95 at k = 5 and 0.45 at
            # k = 6, while the same k = 6 sat at 0.61 when it was interior to a
            # [1, 10] sweep — the same solution, a quarter worse for standing on
            # the edge.
            #
            # It also matters for the input rescaling, which maps [min, max] to
            # [-1, 1]: without this the network never sees an input of exactly
            # +-1, which is the range its initialisation is designed for.
            #
            # Two slots per axis, taken from the random ones. With one swept
            # parameter that is the two endpoints; with several it puts one
            # setting on each face of the box rather than trying to cover its
            # 2^d corners, which would consume the whole batch by four
            # parameters.
            if anchor_ends and n >= 2 * n_axes:
                i = axis_index * 2
                vals[i], vals[i + 1] = lo, hi

            for i, v in enumerate(vals.tolist()):
                new_params[i][key] = v

        for p in new_params:
            for key in self.param_order:
                if key not in self.limits:
                    p[key] = self.par[key][0].item()
        return new_params

    def _resample_parameters(self) -> None:
        """
        Resamples only parameters that have limits defined.
        All limit entries must share the same N (batch size).
        """
        new_params = self.draw_parameters()
        if not new_params:
            return
        self.par = self.make_param_batch(new_params)
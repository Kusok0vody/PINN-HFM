import torch
from physics.phys import Physics
from utils import derivative_batched, unpack_coords
from geometry.sampler import BoundaryBatch


class helmholtz2D_annulus(Physics):
    """
    2D Helmholtz-type equation in an annulus:
        laplace(u) + k u = 0

    Variables:
        u: scalar field

    Parameters:
        k: reaction / wave parameter
    """

    has_reference = True

    def __init__(self, device="cpu", dim=2, has_time=False,
                 r_in: float = 1.0, r_out: float = 3.0, n_arcs: int = 8):
        super().__init__(device, dim, has_time)
        # Geometry of the annulus the closed form belongs to. The boundaries
        # dict passed to setParameters describes the same thing, but as arc
        # parameterisations that would have to be reverse-engineered into radii;
        # asking for them is cheaper and cannot be wrong by inference.
        self.r_in, self.r_out, self.n_arcs = r_in, r_out, n_arcs

    def reference(self, coords, params) -> dict:
        """
        Exact solution of this annulus problem, from the Bessel cross-product
        series in validation.references.

        Raises for k at or near a Dirichlet eigenvalue, where the boundary value
        problem has no unique solution and there is nothing to be exact about.
        Points outside the annulus come back NaN from the series and are
        rejected by whoever installs the data, which is the intended behaviour:
        a silent extrapolation would be worse than a refusal.
        """
        import numpy as np
        from validation.references import helmholtz_annulus

        c = coords.detach().cpu().numpy()
        y, x = c[:, 0], c[:, 1]          # Geometry order for (has_time=False, dim=2)
        ks = params.detach().cpu().numpy()[:, self.param_order.index("k")]
        cols = [helmholtz_annulus(x, y, float(k), self.r_in, self.r_out,
                                  self.n_arcs) for k in ks]
        return {"u": torch.tensor(np.stack(cols, axis=1), dtype=torch.float32)}

    def setParameters(
        self,
        params: list[dict],
        boundaries: dict,
        initial: dict = None,
        limits: dict = {},
    ):
        # initial is accepted and ignored: the problem is stationary, but every
        # other Physics subclass takes the argument and callers pass it
        # positionally or by name without checking which problem they hold.
        required = {"k"}
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

        self.limits = limits

        self.transforms = {
            "u": lambda u: u
        }

    def residualPDE(self, pred: dict, unpacked: dict) -> dict:
        x = unpacked["x"]
        y = unpacked["y"]

        u = pred["u"]

        k = self.par["k"].unsqueeze(0)

        u_x = derivative_batched(u, x)
        u_y = derivative_batched(u, y)

        u_xx = derivative_batched(u_x, x)
        u_yy = derivative_batched(u_y, y)

        laplace_u = u_xx + u_yy

        return {
            "helmholtz": laplace_u + k * u
        }

    def residualBC(self, pred: dict, coords_bc: dict, batch: BoundaryBatch) -> dict:
        if batch.name not in self.boundaries:
            return {}

        x  = coords_bc["x"]
        y  = coords_bc["y"]
        nx = batch.nx
        ny = batch.ny

        u = pred["u"]

        residuals = {}

        for var_name, cond in self.boundaries[batch.name].items():
            key = f"{batch.name}_{var_name}"

            btype = cond["type"]
            value = cond.get("value", None)

            val = value(x, y) if value is not None else None

            if var_name == "u":
                if btype == "dirichlet":
                    residuals[key] = u - val

                elif btype == "neumann":
                    u_x = derivative_batched(u, x)
                    u_y = derivative_batched(u, y)

                    u_n = u_x * nx + u_y * ny

                    target = val if val is not None else torch.zeros_like(u_n)

                    residuals[key] = u_n - target

        return residuals
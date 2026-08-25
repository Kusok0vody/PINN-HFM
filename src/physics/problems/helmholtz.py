import math

import torch
from physics.phys import Physics
from utils import derivative_batched, unpack_coords
from geometry.sampler import BoundaryBatch


def harmonic_annulus(rho, theta, r_in, r_out, n_arcs=8, n_terms=16):
    """
    Harmonic function on the annulus with the arc boundary data, in torch.

    Solves laplace(G) = 0 with G = 0 on rho = r_in and G = sgn(sin(m theta)) on
    rho = r_out, truncated to n_terms modes. Separation of variables gives power
    laws rather than Bessel functions, so every term is elementary:

        H_n(rho) = [(rho/r_out)^n - (r_in^2 / (rho r_out))^n] / [1 - (r_in/r_out)^(2n)]

    which is 0 at r_in and exactly 1 at r_out. The square wave contributes only
    orders n = m, 3m, 5m, ..., with m = n_arcs / 2, and coefficients 4/(pi j).

    Written with every base below one on purpose. The textbook form
    (rho/r_in)^n - (r_in/rho)^n overflows float32 by n = 20 on a 1:3 annulus,
    where (3)^20 is 3.5e9, and the same factor cancels out of the ratio anyway.

    Why harmonic and not, say, a smooth interpolation of the boundary data: this
    G enters a residual that differentiates it twice, and for a harmonic G that
    second derivative is exactly zero. Nothing has to be computed, nothing can
    be inaccurate, and the discontinuity of the boundary datum — which has
    unbounded derivatives near its jumps — never reaches the residual at all.
    The truncation is what is paid instead: the trace on the outer ring is the
    partial Fourier sum, which rings near the jumps by a fixed amount that no
    amount of training would have removed either.
    """
    m = n_arcs // 2
    q = r_in / r_out
    g = torch.zeros_like(rho)
    for j in range(1, 2 * n_terms, 2):
        n = m * j
        a = (rho / r_out) ** n
        b = (r_in * r_in / (rho * r_out)) ** n
        g = g + (4.0 / math.pi) * (a - b) / (1.0 - q ** (2 * n)) * torch.sin(n * theta) / j
    return g


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
                 r_in: float = 1.0, r_out: float = 3.0, n_arcs: int = 8,
                 hard_bc: bool = False, hard_bc_terms: int = 16):
        super().__init__(device, dim, has_time)
        # Geometry of the annulus the closed form belongs to. The boundaries
        # dict passed to setParameters describes the same thing, but as arc
        # parameterisations that would have to be reverse-engineered into radii;
        # asking for them is cheaper and cannot be wrong by inference.
        self.r_in, self.r_out, self.n_arcs = r_in, r_out, n_arcs

        # Imposing the boundary by construction rather than by penalty. Both
        # conditions are homogeneous in the network: u = G + D N with G carrying
        # the data and D vanishing on both rings, so every N satisfies them and
        # sixteen loss terms disappear. That number is the reason to want this
        # here: the balancing rules in the literature are written for two to
        # four terms, and a boundary described as arcs turns one condition into
        # eight.
        self.hard_bc = hard_bc
        self.hard_bc_terms = hard_bc_terms
        if hard_bc:
            self.output_ansatz = {"u": self._hard_bc}

    def _hard_bc(self, u, coords):
        x, y = coords["x"], coords["y"]
        rho = torch.sqrt(x * x + y * y)
        th = torch.atan2(y, x)
        g = harmonic_annulus(rho, th, self.r_in, self.r_out, self.n_arcs,
                             self.hard_bc_terms)
        # Zero on both rings, 1 at mid-radius for the 1:3 annulus, and analytic
        # everywhere between. Nothing about it depends on the parameter, so the
        # same window serves every setting of the sweep.
        d = (rho - self.r_in) * (self.r_out - rho)
        return g + d * u

    # Largest solution the sweep is allowed to draw, for a boundary datum of
    # size one. Above this the problem is not hard but pointless: the boundary
    # condition asks for +-1 while the solution is hundreds of times that, no
    # network of this size represents it, and its residual and data terms
    # dominate every other setting in the batch. Measured on this annulus, a
    # single drawn setting at k = 6.516 has max|u| = 986 against 1.0 at k = 1.
    #
    # 50 is chosen against what the network reaches rather than against the
    # equation: runs here top out around an amplitude of 10, so 50 leaves room
    # to be wrong about that by a factor of five while still excluding the
    # settings that are only ever noise.
    AMPLIFICATION_MAX = 50.0

    def parameter_valid(self, params):
        from validation.references import helmholtz_annulus_amplification

        i = self.param_order.index("k")
        ks = params.detach().cpu().numpy()[:, i]
        amp = [helmholtz_annulus_amplification(float(k), self.r_in, self.r_out,
                                               self.n_arcs) for k in ks]
        return torch.tensor([a < self.AMPLIFICATION_MAX for a in amp],
                            dtype=torch.bool)

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
        # Under the hard ansatz the conditions hold identically, so a penalty
        # for them would be a term that is zero by construction and still
        # consumes a share of the balancing.
        if self.hard_bc or batch.name not in self.boundaries:
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
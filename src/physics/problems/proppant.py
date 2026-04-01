import torch
from src.physics.phys import Physics
from src.physics.parameters import *
from src.utils import derivative_batched, unpack_coords, smooth_clamp
from src.geometry.sampler import BoundaryBatch


class proppantDynamics_dless(Physics):
    """
    Undimensionalised statement for proppant dynamics in opened fracture.

    Variables:
        c  — proppant concentration, c in [0, 1)
        px — dp/dx
        py — dp/dy

    PDE:
        dc/dt - w^2/mu * px * dc/dx - alpha^2 * w^2/mu * (py - (1+rc)G) * dc/dy = 0
        d/dx(w^3/mu * px) + d/dy(w^3/mu * (py - (1+rc)G)) = 0
        mu = (1 - c)^beta

    Boundary conditions:
        inlet:  c = c_in(t),  dp/dn = -mu/w^2 * u_in
        outlet: du/dn = 0  →  dp/dn = 0
        slip:   u = 0      →  dp/dn = alpha * (1 + rc) * G * e_y

    Initial condition:
        c = 0

    Parameters (each is a tensor of shape (M,) for M parameter settings):
        alpha: aspect ratio L/H
        beta:  viscosity exponent
        r:     (rho_p - rho_f) / rho_f * c_max
        G:     rho_f * g * H / p_0
        w:     fracture width, callable w(x, y) -> (N, 1)
        mu:    viscosity model, callable mu(c) -> (N, M)
    """

    def __init__(self, device="cpu", dim=2, has_time=True):
        super().__init__(device, dim, has_time)

    def setParameters(self, params: list[dict], funcPar: dict, boundaries: dict, initial: dict = None):
        required = {"alpha", "beta", "r", "G"}
        missing  = required - params[0].keys()
        if missing:
            raise ValueError(f"Missing parameters: {missing}")

        self.param_order = list(params[0].keys())
        self.par = self.make_param_batch(params)

        self.boundaries = {
            name: bound["bc"]
            for name, bound in boundaries.items()
            if "bc" in bound
        }
        self.initial    = initial or {
            "c": lambda x, y: torch.zeros_like(x)
        }
        self.transforms = {
            "c": lambda c: smooth_clamp(torch.sigmoid(c), lo=0.0, hi=0.999, eps=1e-4),
        }
        self.funcPar = funcPar or {
            "w": lambda x, y: torch.ones_like(x)
        }

    def _viscosity(self, c: torch.Tensor) -> torch.Tensor:
        """mu = (1 - c)^beta"""
        beta = self.par["beta"].unsqueeze(0)   # (1, M)
        return (1 - c) ** beta

    def residualPDE(self, pred: dict, unpacked: dict) -> dict:
        t = unpacked["t"]
        x = unpacked["x"]
        y = unpacked["y"]

        c   = pred["c"]    # (N, M)
        p_x = pred["px"]   # (N, M)
        p_y = pred["py"]   # (N, M)

        alpha = self.par["alpha"].unsqueeze(0)   # (1, M)
        r     = self.par["r"].unsqueeze(0)       # (1, M)
        G     = self.par["G"].unsqueeze(0)       # (1, M)

        width = self.funcPar["w"](x, y)          # (N, 1)
        mu    = self._viscosity(c)               # (N, M)
        gravity  = (1 + r * c) * G               # (N, M)
        mobility = width**2 / mu                 # (N, M)

        ux = -mobility * p_x                     # (N, M)
        uy = -mobility * (p_y - gravity) * alpha # (N, M)

        c_t = derivative_batched(c, t)
        c_x = derivative_batched(c, x) * ux
        c_y = derivative_batched(c, y) * uy * alpha

        ux_x = derivative_batched(width * ux, x)
        uy_y = derivative_batched(width * uy, y) * alpha

        return {
            "convection":  c_t + c_x + c_y,
            "poisson":     ux_x + uy_y,
            "correlation": derivative_batched(p_x, y) - derivative_batched(p_y, x),
        }

    def residualBC(self, pred: dict, coords_bc: torch.Tensor, batch: BoundaryBatch) -> dict:
        if batch.name not in self.boundaries:
            return {}

        unpacked = unpack_coords(coords_bc, self.has_time, self.dim)

        t  = unpacked.get("t")
        x  = unpacked["x"]
        y  = unpacked["y"]
        nx = batch.nx      # (N, 1)
        ny = batch.ny      # (N, 1)

        c   = pred["c"]    # (N, M)
        p_x = pred["px"]   # (N, M)
        p_y = pred["py"]   # (N, M)

        width = self.funcPar["w"](x, y)          # (N, 1)
        mu    = self._viscosity(c)               # (N, M)

        alpha = self.par["alpha"].unsqueeze(0)   # (1, M)
        r     = self.par["r"].unsqueeze(0)       # (1, M)
        G     = self.par["G"].unsqueeze(0)       # (1, M)

        gravity = (1 + r * c) * G                # (N, M)
        p_n     = p_x * nx + p_y * ny            # (N, M)

        residuals = {}

        for var_name, cond in self.boundaries[batch.name].items():
            key   = f"{batch.name}_{var_name}"
            btype = cond["type"]
            value = cond.get("value", None)
            val   = value(t, x, y) if value is not None else None

            if var_name == "c":
                if btype == "dirichlet":
                    residuals[key] = c - val
                elif btype == "neumann":
                    c_n = derivative_batched(c, x) * nx + derivative_batched(c, y) * ny
                    residuals[key] = c_n - (val if val is not None else torch.zeros_like(c_n))

            elif var_name == "u":
                mobility = width**2 / mu                 # (N, M)
                ux  = mobility * p_x                     # (N, M)
                uy  = mobility * (p_y - gravity) * alpha # (N, M)
                u_n = ux  * nx + uy  * ny                # (N, M)
                
                if btype == "dirichlet":
                    # u_n = val
                    residuals[key] = u_n - (value(t, x, y) if value else torch.zeros_like(u_n))
                elif btype == "neumann":
                    # du/dn = val (0 for free outlet)
                    residuals[key] = u_n - (value(t, x, y) if value else torch.zeros_like(u_n))

            elif var_name == "p":
                if btype == "dirichlet":
                    residuals[key] = p_n - (val if val is not None else torch.zeros_like(p_n))
                elif btype == "neumann":
                    # dp/dn = -mu/w^2 * u_in*nx + gravity*ny
                    u_in = val if val is not None else torch.zeros_like(p_n)
                    residuals[key] = p_n - (-mu / width**2 * u_in * nx + gravity * ny)

        return residuals
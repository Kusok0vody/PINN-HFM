import math
import torch

from src.physics.phys import Physics
from src.utils import derivative_batched
from src.geometry.sampler import BoundaryBatch


class ElasticWave2D(Physics):
    """
    2D elastic wave equation in velocity-stress form on a rectangle.

    Variables:
        vx, vy        — velocity components
        sxx, syy, sxy — stress tensor components (sxy = syx)

    PDE:
        rho * dvx/dt - (dsxx/dx + dsxy/dy) = fx
        rho * dvy/dt - (dsxy/dx + dsyy/dy) = fy
        dsxx/dt - ((lam + 2*mu) * dvx/dx + lam * dvy/dy)         = 0
        dsyy/dt - (lam * dvx/dx + (lam + 2*mu) * dvy/dy)         = 0
        dsxy/dt - (mu * (dvx/dy + dvy/dx))                       = 0
        lam = rho * (Vp^2 - 2*Vs^2),  mu = rho * Vs^2

    A sponge layer of width L_pml adds damping d(x, y) to every equation
    (rho*d*v and d*sigma terms), zero in the interior, rising to d_max at
    the rectangle edges. This approximates a PML without field splitting.

    Source (explosive, f = -grad(F(t) * g(r))):
        F(t) = -2*pi^2 * f0^2 * (t - t0) * exp(-pi^2 * f0^2 * (t - t0)^2),  t < 2*t0
        g(r) = (1 - r^2/a^2)^3,  r < a,  a = 5*h
        fx = 6 * F(t) * (x - xs) / a^2 * (1 - r^2/a^2)^2
        fy = 6 * F(t) * (y - ys) / a^2 * (1 - r^2/a^2)^2

    Boundary conditions:
        dirichlet / neumann per variable, or "traction" for sigma*n = (tx, ty)

    Initial condition:
        vx = vy = sxx = syy = sxy = 0

    Parameters (each is a tensor of shape (M,) for M parameter settings):
        f0: Ricker wavelet central frequency

    setParameters keyword arguments:
        medium: {"rho": fn(x,y), "Vp": fn(x,y), "Vs": fn(x,y)}
        source: {"xs": float, "ys": float, "t0": float, "h": float}
        domain: {"x_min", "x_max", "y_min", "y_max"}
        pml:    {"thickness": L_pml, "damping": d_max, "power": p}, optional
    """

    def __init__(self, device="cpu", dim=2, has_time=True):
        super().__init__(device, dim, has_time)

    @staticmethod
    def _smooth_relu(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Smooth max(z, 0) with a live gradient everywhere."""
        return 0.5 * (z + torch.sqrt(z * z + eps * eps))

    @staticmethod
    def _smooth_step(z: torch.Tensor, tau: float) -> torch.Tensor:
        """Smooth indicator of z < 0."""
        return 0.5 * (1.0 - torch.tanh(z / tau))

    def setParameters(self, params: list[dict], boundaries: dict,
                      medium: dict, source: dict, domain: dict,
                      pml: dict = None, initial: dict = None, limits: dict = {}):
        required = {"f0"}
        missing  = required - params[0].keys()
        if missing:
            raise ValueError(f"Missing parameters: {missing}")

        self.param_order = list(params[0].keys())
        self.par = self.make_param_batch(params)
        self.limits = limits

        for k in ("rho", "Vp", "Vs"):
            if k not in medium or not callable(medium[k]):
                raise ValueError(f"medium['{k}'] must be a callable(x, y)")
        self.medium = medium

        for k in ("xs", "ys", "t0", "h"):
            if k not in source:
                raise ValueError(f"source['{k}'] is required")
        self.xs = float(source["xs"])
        self.ys = float(source["ys"])
        self.t0 = float(source["t0"])
        self.h  = float(source["h"])
        self.a  = 5.0 * self.h

        for k in ("x_min", "x_max", "y_min", "y_max"):
            if k not in domain:
                raise ValueError(f"domain['{k}'] is required")
        self.x_min = float(domain["x_min"])
        self.x_max = float(domain["x_max"])
        self.y_min = float(domain["y_min"])
        self.y_max = float(domain["y_max"])

        pml = pml or {}
        self.pml_thickness = float(pml.get("thickness", 0.0))
        self.pml_damping   = float(pml.get("damping", 0.0))
        self.pml_power     = float(pml.get("power", 2.0))

        self.boundaries = {
            name: {
                var: cond
                for var, cond in bound.get("bc", {}).items()
                if isinstance(cond, dict) and "type" in cond
            }
            for name, bound in boundaries.items()
            if bound.get("bc")
        }
        self.boundary_meta = {
            name: {
                "periodic": bound.get("periodic", False),
                "with":     bound.get("with", None),
            }
            for name, bound in boundaries.items()
        }

        zero_xy = lambda x, y: torch.zeros_like(x)
        self.initial = initial or {
            "vx":  zero_xy, "vy":  zero_xy,
            "sxx": zero_xy, "syy": zero_xy, "sxy": zero_xy,
        }
        self.transforms = {}

    def _pml_damping(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Sponge damping d(x, y), zero in the interior."""
        if self.pml_thickness <= 0.0 or self.pml_damping <= 0.0:
            return torch.zeros_like(x)

        L   = self.pml_thickness
        p   = self.pml_power
        d0  = self.pml_damping
        eps = 1e-6 * L

        dist_x = (self._smooth_relu(self.x_min + L - x, eps)
                  + self._smooth_relu(x - (self.x_max - L), eps))
        dist_y = (self._smooth_relu(self.y_min + L - y, eps)
                  + self._smooth_relu(y - (self.y_max - L), eps))

        return d0 * ((dist_x / L).pow(p) + (dist_y / L).pow(p))

    def _source(self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
        """Explosive source f = -grad(F(t) * g(r)), returns (fx, fy) of shape (N, M)."""
        f0  = self.par["f0"].unsqueeze(0)
        t0  = self.t0
        pi2 = math.pi ** 2

        arg = pi2 * f0 ** 2 * (t - t0) ** 2
        tau = max(t0 / 40.0, 1e-6)
        win = self._smooth_step(t - 2.0 * t0, tau)
        F_t = -2.0 * pi2 * f0 ** 2 * (t - t0) * torch.exp(-arg) * win

        dx = x - self.xs
        dy = y - self.ys
        r2 = dx * dx + dy * dy
        a2 = self.a * self.a

        bump = self._smooth_relu(1.0 - r2 / a2, eps=1e-6)
        coef = (6.0 / a2) * bump.pow(2)

        return F_t * (coef * dx), F_t * (coef * dy)

    def residualPDE(self, pred: dict, unpacked: dict) -> dict:
        t = unpacked["t"]
        x = unpacked["x"]
        y = unpacked["y"]

        vx,  vy  = pred["vx"],  pred["vy"]
        sxx, syy = pred["sxx"], pred["syy"]
        sxy      = pred["sxy"]

        vx_t  = derivative_batched(vx,  t)
        vy_t  = derivative_batched(vy,  t)
        sxx_t = derivative_batched(sxx, t)
        syy_t = derivative_batched(syy, t)
        sxy_t = derivative_batched(sxy, t)

        vx_x = derivative_batched(vx, x)
        vx_y = derivative_batched(vx, y)
        vy_x = derivative_batched(vy, x)
        vy_y = derivative_batched(vy, y)

        sxx_x = derivative_batched(sxx, x)
        sxy_y = derivative_batched(sxy, y)
        sxy_x = derivative_batched(sxy, x)
        syy_y = derivative_batched(syy, y)

        rho = self.medium["rho"](x, y)
        Vp  = self.medium["Vp"](x, y)
        Vs  = self.medium["Vs"](x, y)
        mu  = rho * Vs.pow(2)
        lam = rho * (Vp.pow(2) - 2.0 * Vs.pow(2))

        d = self._pml_damping(x, y)
        fx, fy = self._source(t, x, y)

        eps_xx = vx_x
        eps_yy = vy_y
        eps_xy = 0.5 * (vx_y + vy_x)

        return {
            "mom_x":   rho * vx_t + rho * d * vx - (sxx_x + sxy_y) - fx,
            "mom_y":   rho * vy_t + rho * d * vy - (sxy_x + syy_y) - fy,
            "cons_xx": sxx_t + d * sxx - ((lam + 2.0 * mu) * eps_xx + lam * eps_yy),
            "cons_yy": syy_t + d * syy - (lam * eps_xx + (lam + 2.0 * mu) * eps_yy),
            "cons_xy": sxy_t + d * sxy - (2.0 * mu * eps_xy),
        }

    def residualBC(self, pred: dict, coords_bc: dict, batch: BoundaryBatch) -> dict:
        if batch.name not in self.boundaries:
            return {}

        t  = coords_bc["t"]
        x  = coords_bc["x"]
        y  = coords_bc["y"]
        nx = batch.nx
        ny = batch.ny

        residuals = {}
        for var_name, cond in self.boundaries[batch.name].items():
            btype = cond["type"]

            if btype == "traction":
                sxx, syy, sxy = pred["sxx"], pred["syy"], pred["sxy"]
                tx_fn = cond.get("tx", lambda t, x, y: torch.zeros_like(x))
                ty_fn = cond.get("ty", lambda t, x, y: torch.zeros_like(x))
                residuals[f"{batch.name}_traction_x"] = sxx * nx + sxy * ny - tx_fn(t, x, y)
                residuals[f"{batch.name}_traction_y"] = sxy * nx + syy * ny - ty_fn(t, x, y)
                continue

            if var_name not in pred:
                continue
            u   = pred[var_name]
            key = f"{batch.name}_{var_name}"
            value = cond.get("value", None)
            val   = value(t, x, y) if value is not None else None

            if btype == "dirichlet":
                residuals[key] = u - (val if val is not None else torch.zeros_like(u))
            elif btype == "neumann":
                u_n = derivative_batched(u, x) * nx + derivative_batched(u, y) * ny
                residuals[key] = u_n - (val if val is not None else torch.zeros_like(u_n))

        return residuals

import torch
import torch.nn.functional as F

from physics.phys import Physics
from utils import derivative_batched
from geometry.sampler import BoundaryBatch


class LotkaVolterra(Physics):
    """
    Three-species Lotka-Volterra system (grass G, herbivores H, predators P).

    Implemented as a time-free 1D problem (dim=1, has_time=False): the t axis
    is treated as the "x" coordinate, the Cauchy initial data is imposed on the
    left boundary (x=0), the right boundary (x=T) is free. Outputs are passed
    through softplus to keep G, H, P strictly positive.

    Variables:
        G — grass biomass
        H — herbivore population
        P — predator population

    PDE:
        dG/dt = r * G * (1 - G/K) - alpha * G * H
        dH/dt = e1 * alpha * G * H - d1 * H - beta * H * P
        dP/dt = e2 * beta * H * P - d2 * P

    Boundary conditions:
        dirichlet on x=0 (initial values of G, H, P)

    Initial condition:
        imposed via the x=0 boundary

    Parameters (each is a tensor of shape (M,) for M parameter settings):
        r     — intrinsic growth rate of grass
        K     — carrying capacity of grass
        alpha — grazing rate (consumption of G by H)
        e1    — conversion efficiency of grass into herbivore biomass
        d1    — death rate of herbivores
        beta  — predation rate (consumption of H by P)
        e2    — conversion efficiency of herbivores into predator biomass
        d2    — death rate of predators
    """

    # Names of the initial values when they are swept as parameters rather
    # than fixed. Kept in one place because three string literals spread over
    # the reference, the ansatz and the validation is three places to misspell.
    IC_PARAMS = {"G": "G0", "H": "H0", "P": "P0"}

    def __init__(self, device="cpu", dim=1, has_time=False, hard_ic=False):
        super().__init__(device, dim, has_time)
        self.hard_ic = hard_ic

    def setParameters(
        self,
        params: list[dict],
        boundaries: dict,
        initial: dict = None,
        limits: dict = {},
    ):
        required = {"r", "K", "alpha", "e1", "d1", "beta", "e2", "d2"}
        missing  = required - set(params[0].keys())
        if missing:
            raise ValueError(f"Missing parameters: {missing}")

        # The Cauchy data can be swept like anything else. When it is, it lives
        # in the parameter vector and every setting carries its own; when it is
        # not, it is a constant read off the boundary specification.
        ic_keys = set(self.IC_PARAMS.values())
        given = ic_keys & set(params[0].keys())
        self.ic_swept = given == ic_keys
        if given and not self.ic_swept:
            raise ValueError(
                "initial values must be swept together or not at all; got "
                + ", ".join(sorted(given)) + " of " + ", ".join(sorted(ic_keys))
            )

        self.param_order = list(params[0].keys())
        self.par         = self.make_param_batch(params)
        self.limits      = limits

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

        self.initial = initial or {}

        if self.hard_ic:
            if not self.ic_swept:
                raise ValueError(
                    "hard_ic needs the initial values in the parameter vector ("
                    + ", ".join(sorted(self.IC_PARAMS.values()))
                    + "); with them fixed the ansatz would bake one Cauchy "
                    "datum into the network and the sweep would not be over "
                    "initial conditions at all"
                )
            # Positivity moves out of the transform and into the ansatz, which
            # has to apply softplus last anyway to hit the initial value
            # exactly. The entries stay, as identities, because this dict is
            # also how the rest of the framework learns which variables this
            # problem defines.
            self.transforms = {n: (lambda u: u) for n in self.IC_PARAMS}
            self.output_ansatz = {n: self._hard_ic(n) for n in self.IC_PARAMS}
        else:
            self.transforms = {
                "G": lambda u: F.softplus(u),
                "H": lambda u: F.softplus(u),
                "P": lambda u: F.softplus(u),
            }

    has_reference = True

    def initial_values(self) -> dict:
        """
        The Cauchy data for the current batch, as {name: (M,) tensor}.

        Two sources, never both. Swept, it is in the parameter vector and each
        setting carries its own. Fixed, it is read back out of the boundary
        specification, where it was already declared once as the Dirichlet
        condition — a second copy passed in separately would be a second thing
        to keep in step. The boundary is identified by carrying a Dirichlet
        condition for every variable, which is what makes it the Cauchy face
        rather than a face that happens to constrain one species.
        """
        if self.ic_swept:
            return {n: self.par[key].reshape(-1)
                    for n, key in self.IC_PARAMS.items()}

        wanted = set(self.IC_PARAMS)
        faces = [name for name, conds in self.boundaries.items()
                 if wanted <= {v for v, c in conds.items()
                               if c.get("type") == "dirichlet"}]
        if len(faces) != 1:
            raise ValueError(
                "the reference needs exactly one boundary carrying Dirichlet "
                "data for all of " + ", ".join(sorted(wanted)) + ", to read "
                "the initial values from; found " + (str(faces) or "none")
            )
        zero = torch.zeros(1, 1, device=self.par.tensor.device)
        m = self.par.tensor.shape[0]
        return {v: self.boundaries[faces[0]][v]["value"](zero).reshape(1).expand(m)
                for v in wanted}

    def _timescale(self):
        """
        The system's own time constant per setting, (M,).

        The ansatz needs to know how quickly the initial value stops dictating
        the solution. Reading it off the equations rather than picking a number
        keeps it out of the list of things to tune, and makes it right for a
        setting whose rates are ten times another's.
        """
        p = self.par
        scale = p["K"].reshape(-1)
        rate = torch.stack([
            p["r"].reshape(-1), p["d1"].reshape(-1), p["d2"].reshape(-1),
            p["alpha"].reshape(-1) * scale,
            p["e1"].reshape(-1) * p["alpha"].reshape(-1) * scale,
            p["beta"].reshape(-1) * scale,
            p["e2"].reshape(-1) * p["beta"].reshape(-1) * scale,
        ]).max(dim=0).values
        return 1.0 / rate.clamp_min(1e-8)

    def _hard_ic(self, name):
        """Build the ansatz for one variable: exact Cauchy data, and positive."""
        def ansatz(u, coords):
            t  = coords["x"]
            u0 = self.initial_values()[name].clamp_min(1e-8)
            # Inverse softplus, written so a large u0 does not overflow exp
            # before the log undoes it.
            a   = u0 + torch.log(-torch.expm1(-u0))
            tau = self._timescale()

            if u.dim() == 2 and u.shape[-1] == a.numel():
                a, tau = a.reshape(1, -1), tau.reshape(1, -1)

            # 1 - exp(-t/tau), not t.
            #
            # Both vanish at t = 0, which is all the ansatz strictly needs, but
            # a bare t keeps growing: at the far end of a thirty-unit horizon
            # it would multiply the network's output by thirty, and to produce
            # an ordinary solution near t = 0 the network would have to output
            # something going like 1/t. This saturates at one over the system's
            # own timescale, so the output is the same size everywhere and has
            # no singular profile to imitate.
            d = -torch.expm1(-t / tau)
            return F.softplus(a + d * u)
        return ansatz

    def reference(self, coords, params) -> dict:
        """
        Reference solution, integrated. See validation.references.lotka_volterra.

        There is no closed form, so "reference" here means a numerical solution
        far more accurate than anything being measured against it: the
        interpolated error is around 1e-11 over the sweep's ranges, against
        network errors of a per cent.
        """
        import numpy as np
        from validation.references import lotka_volterra

        t   = coords.detach().cpu().numpy().reshape(-1)
        mu  = params.detach().cpu().numpy()
        par = {k: mu[:, self.param_order.index(k)]
               for k in ("r", "K", "alpha", "e1", "d1", "beta", "e2", "d2")}
        y0  = self.initial_values()
        y0  = np.stack([y0[n].detach().cpu().numpy() for n in ("G", "H", "P")],
                       axis=1)                      # (M, 3), one row per setting

        y = lotka_volterra(t, y0, par)
        return {name: torch.tensor(y[:, :, j], dtype=torch.float32)
                for j, name in enumerate(("G", "H", "P"))}

    def residualPDE(self, pred: dict, unpacked: dict) -> dict:
        t = unpacked["x"]

        G = pred["G"]
        H = pred["H"]
        P = pred["P"]

        r     = self.par["r"].unsqueeze(0)
        K     = self.par["K"].unsqueeze(0)
        alpha = self.par["alpha"].unsqueeze(0)
        e1    = self.par["e1"].unsqueeze(0)
        d1    = self.par["d1"].unsqueeze(0)
        beta  = self.par["beta"].unsqueeze(0)
        e2    = self.par["e2"].unsqueeze(0)
        d2    = self.par["d2"].unsqueeze(0)

        dG = derivative_batched(G, t)
        dH = derivative_batched(H, t)
        dP = derivative_batched(P, t)

        return {
            "G_eq": dG - (r * G * (1 - G / K) - alpha * G * H),
            "H_eq": dH - (e1 * alpha * G * H - d1 * H - beta * H * P),
            "P_eq": dP - (e2 * beta * H * P - d2 * P),
        }

    def residualBC(self, pred: dict, coords_bc: dict, batch: BoundaryBatch) -> dict:
        # With the ansatz in place the Cauchy data holds identically, so a
        # residual for it would be zero by construction: a loss term that
        # cannot be reduced because it is already zero, competing for weight
        # against the ones that can.
        if self.hard_ic or batch.name not in self.boundaries:
            return {}

        x = coords_bc["x"]

        residuals = {}
        for var_name, cond in self.boundaries[batch.name].items():
            if cond["type"] == "dirichlet":
                val = cond["value"](x)
                residuals[f"{batch.name}_{var_name}"] = pred[var_name] - val

        return residuals

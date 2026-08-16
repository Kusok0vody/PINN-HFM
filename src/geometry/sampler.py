import torch
from dataclasses import dataclass, field

from geometry.geom import *

@dataclass
class BoundaryBatch:
    """Points sampled on a single boundary segment."""
    coords: torch.Tensor   # (n, n_coords)
    nx:     torch.Tensor   # (n, 1)  — frozen outward normal, no grad
    ny:     torch.Tensor   # (n, 1)
    name:   str = ""
    
    def to(self, device) -> "BoundaryBatch":
        return BoundaryBatch(
            coords=self.coords.to(device),
            nx=self.nx.to(device),
            ny=self.ny.to(device),
            name=self.name,
        )

@dataclass
class CollocationBatch:
    """Interior (or initial-condition) collocation points."""
    coords: torch.Tensor   # (n, n_coords)
    
    def to(self, device) -> "CollocationBatch":
        return CollocationBatch(coords=self.coords.to(device))


@dataclass
class SampledPoints:
    """
    Full set of points produced by Sampler.sample().
    Stored inside PINN and consumed by Physics when computing loss.
    """
    interior:   CollocationBatch
    boundaries: dict = field(default_factory=dict)   # str -> BoundaryBatch
    initial:    CollocationBatch | None = None       # only when has_time=True

    def to(self, device) -> "SampledPoints":
        return SampledPoints(
            interior=self.interior.to(device),
            boundaries={name: b.to(device) for name, b in self.boundaries.items()},
            initial=self.initial.to(device) if self.initial is not None else None,
        )


class Sampler:
    """
    Generates collocation and boundary points for PINN training.

    Geometry describes the domain; Sampler decides *how* to populate it.
    Decoupling these two makes it straightforward to swap sampling strategies
    (uniform, Latin-hypercube, adaptive residual-based) without touching Geometry.

    Args:
        geometry:       a Geometry instance
        n_interior:     number of interior collocation points
        n_boundary:     number of points per boundary segment
        n_initial:      number of initial-condition points (only used when has_time=True)
        strategy:       "uniform" (default) or "latin_hypercube"
        rejection_max:  maximum candidates drawn per accepted interior point
                        during rejection sampling (uniform strategy only)
    """

    STRATEGIES = ("uniform", "latin_hypercube")

    def __init__(
        self,
        geometry: Geometry,
        n_interior:    int = 1000,
        n_boundary:    int = 200,
        n_initial:     int = 500,
        strategy:      str = "uniform",
        rejection_max: int = 10,
    ):
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Choose from {self.STRATEGIES}."
            )
        self.geometry      = geometry
        self.n_interior    = n_interior
        self.n_boundary    = n_boundary
        self.n_initial     = n_initial
        self.strategy      = strategy
        self.rejection_max = rejection_max

    def sample(self) -> SampledPoints:
        """
        Draw a fresh set of points.  Call this once per training epoch
        (or less frequently for static domains).

        Returns:
            SampledPoints with interior, boundaries, and (optionally) initial
        """
        interior   = CollocationBatch(coords=self._sample_interior())
        boundaries = self._sample_boundaries()
        initial    = (
            CollocationBatch(coords=self._sample_initial())
            if self.geometry.has_time
            else None
        )
        return SampledPoints(
            interior=interior,
            boundaries=boundaries,
            initial=initial,
        )

    def _sample_interior(self) -> torch.Tensor:
        if self.strategy == "uniform":
            return self._uniform_interior(self.n_interior)
        return self._lhs_interior(self.n_interior)

    def _uniform_interior(self, n: int) -> torch.Tensor:
        """
        Rejection sampling: draw candidates uniformly from the bounding box,
        keep only those inside the domain.
        """
        geo        = self.geometry
        collected  = []
        remaining  = n
        
        x_min, x_max = self._x_bounds()
        y_min, y_max = self._y_bounds() if geo.dim == 2 else (None, None)

        while remaining > 0:
            batch = remaining * self.rejection_max

            coords_parts = []
            if geo.has_time:
                t = torch.empty(batch).uniform_(geo.T[0], geo.T[1])
                coords_parts.append(t)
            if geo.dim == 2:
                y = torch.empty(batch).uniform_(y_min, y_max)
                coords_parts.append(y)
            x = torch.empty(batch).uniform_(x_min, x_max)
            coords_parts.append(x)

            candidates = torch.stack(coords_parts, dim=1)
            mask       = geo.inside(candidates)
            accepted   = candidates[mask]

            collected.append(accepted[:remaining])
            remaining -= accepted.shape[0]
            remaining  = max(remaining, 0)

        return torch.cat(collected, dim=0)[:n]

    def _lhs_interior(self, n: int) -> torch.Tensor:
        """
        Latin Hypercube Sampling in the bounding box, then reject points
        outside the domain (and top up with uniform draws if needed).
        """
        geo = self.geometry
        n_coords = len(geo.coord_order)

        perms  = [torch.randperm(n) for _ in range(n_coords)]
        unit   = torch.stack(
            [(perms[i] + torch.rand(n)) / n for i in range(n_coords)],
            dim=1,
        )

        lows, highs = self._bounding_box()
        candidates  = unit * (highs - lows) + lows

        mask     = geo.inside(candidates)
        accepted = candidates[mask]

        if accepted.shape[0] < n:
            extra    = self._uniform_interior(n - accepted.shape[0])
            accepted = torch.cat([accepted, extra], dim=0)

        return accepted[:n]

    def _find_periodic_pairs(self):
        pairs = []
        visited = set()

        for name, bound in self.geometry.boundaries.items():
            if bound.get("periodic", False):
                other = bound["with"]
                key = tuple(sorted([name, other]))

                if key not in visited:
                    pairs.append((name, other))
                    visited.add(key)

        return pairs

    def _sample_boundaries(self) -> dict:
        boundaries = {}
        used = set()

        for name_L, name_R in self._find_periodic_pairs():
            bL, bR = self._sample_periodic_pair(name_L, name_R)
            boundaries[name_L] = bL
            boundaries[name_R] = bR
            used.add(name_L)
            used.add(name_R)

        for name, bound in self.geometry.boundaries.items():
            if name in used:
                continue
            boundaries[name] = self._sample_one_boundary(name, bound)

        return boundaries

    def _sample_one_boundary(self, name: str, bound: dict, p=None, t=None) -> BoundaryBatch:
        geo = self.geometry
        n = bound.get("N", self.n_boundary)
        p_min, p_max = bound["p"]

        if n == 0:
            n_coords = len(geo.coord_order)
            return BoundaryBatch(
                coords=torch.zeros(0, n_coords),
                nx=torch.zeros(0, 1),
                ny=torch.zeros(0, 1),
                name=name,
            )

        if p is None:
            p = torch.empty(n).uniform_(p_min, p_max)

        if geo.has_time:
            if t is None:
                t = torch.empty(n).uniform_(geo.T[0], geo.T[1])

        if geo.dim == 2:
            nx, ny = geo.compute_normal(bound, p)
        else:
            x_vals = bound["x"](p).detach()

            eps = 1e-6
            x_test = x_vals + eps

            test_pts = x_test.unsqueeze(1)

            is_inside = geo.inside_spatial(test_pts)

            sign = torch.where(is_inside, -1.0, 1.0).unsqueeze(1)

            nx = sign
            ny = torch.zeros_like(nx)

        parts = []

        if geo.has_time:
            parts.append(t)

        if geo.dim == 2:
            parts.append(bound["y"](p).detach())

        parts.append(bound["x"](p).detach())

        coords = torch.stack(parts, dim=1)

        return BoundaryBatch(coords=coords, nx=nx, ny=ny, name=name)

    def _sample_periodic_pair(self, name_L: str, name_R: str):
        geo = self.geometry
        bound_L = geo.boundaries[name_L]
        bound_R = geo.boundaries[name_R]

        n = bound_L.get("N", self.n_boundary)
        p_min, p_max = bound_L["p"]

        p = torch.empty(n).uniform_(p_min, p_max)

        t = None
        if geo.has_time:
            t = torch.empty(n).uniform_(geo.T[0], geo.T[1])

        bL = self._sample_one_boundary(name_L, bound_L, p=p, t=t)
        bR = self._sample_one_boundary(name_R, bound_R, p=p, t=t)

        return bL, bR

    def _sample_initial(self) -> torch.Tensor:
        """
        Sample points at t = T[0] inside the spatial domain.
        Coordinates are ordered according to geo.coord_order,
        with the time component fixed at T[0].
        """
        geo = self.geometry
        n   = self.n_initial

        spatial_sampler = Sampler(
            geometry=Geometry(
                boundaries=geo.boundaries,
                dim=geo.dim,
                has_time=False,
                T=geo.T,
            ),
            n_interior=n,
            strategy=self.strategy,
            rejection_max=self.rejection_max,
        )
        spatial = spatial_sampler._sample_interior()
        
        t_col = torch.full((n, 1), geo.T[0])
        return torch.cat([t_col, spatial], dim=1)
    
    def _n_boundary(self, name: str) -> int:
        return self.geometry.boundaries[name].get("N", self.n_boundary)
    
    def _x_bounds(self) -> tuple[float, float]:
        geo = self.geometry
        with torch.no_grad():
            vals = torch.cat([
                bound["x"](torch.linspace(*bound["p"], 100))
                for bound in geo.boundaries.values()
            ])
        return vals.min().item(), vals.max().item()

    def _y_bounds(self) -> tuple[float, float]:
        geo = self.geometry
        with torch.no_grad():
            vals = torch.cat([
                bound["y"](torch.linspace(*bound["p"], 100))
                for bound in geo.boundaries.values()
            ])
        return vals.min().item(), vals.max().item()

    def _bounding_box(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (lows, highs) tensors aligned with geo.coord_order."""
        geo   = self.geometry
        lows  = []
        highs = []

        for coord in geo.coord_order:
            if coord == "t":
                lows.append(geo.T[0]);  highs.append(geo.T[1])
            elif coord == "x":
                lo, hi = self._x_bounds();  lows.append(lo);  highs.append(hi)
            elif coord == "y":
                lo, hi = self._y_bounds();  lows.append(lo);  highs.append(hi)

        return torch.tensor(lows), torch.tensor(highs)

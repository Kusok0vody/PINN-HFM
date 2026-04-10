import torch
import math


class Geometry:
    """
    Pure descriptor of the space-time domain.

    Responsible for:
      - knowing the coordinate order convention
      - checking whether a point is inside the domain (inside)
      - computing outward unit normals on boundaries (compute_normal)
      - unpacking a coordinate tensor into a named dict (unpack_coords)

    NOT responsible for sampling — that belongs to Sampler.

    boundaries format:
    {
      "left":  {"p": [a, b], "x": callable, "y": callable,
                "normal_dir": 1},   # normal_dir is optional, default 1
      ...
    }

    Coordinate order convention:
        [t, y, x]  if has_time and dim == 2
        [t, x]     if has_time and dim == 1
        [y, x]     if dim == 2
        [x]        if dim == 1
    """

    COORD_ORDER = {
        (True,  2): ["t", "y", "x"],
        (True,  1): ["t", "x"],
        (False, 2): ["y", "x"],
        (False, 1): ["x"],
    }

    def __init__(
        self,
        boundaries: dict,
        dim: int = 2,
        has_time: bool = False,
        T: list | None = None,
    ):
        self.boundaries  = boundaries
        self.dim         = dim
        self.has_time    = has_time
        self.T           = T if T is not None else [0.0, 1.0]
        self.coord_order = self.COORD_ORDER[(has_time, dim)]
        self._validate()

    def _validate(self):
        for name, bound in self.boundaries.items():
            for key in ("p", "x"):
                if key not in bound:
                    raise ValueError(
                        f"Boundary '{name}' is missing required key '{key}'."
                    )
            if self.dim == 2 and "y" not in bound:
                raise ValueError(
                    f"Boundary '{name}' is missing key 'y' (required for dim=2)."
                )
            p = bound["p"]
            if len(p) != 2 or p[0] >= p[1]:
                raise ValueError(
                    f"Boundary '{name}': 'p' must be [p_min, p_max] with p_min < p_max."
                )
                
    def unpack_coords(self, coords: torch.Tensor) -> dict:
        """
        Unpack a coordinate tensor into a named dict.

        Args:
            coords: (n, n_coords)

        Returns:
            dict, e.g. {"t": (n,1), "y": (n,1), "x": (n,1)}
        """
        return {
            name: coords[:, i : i + 1]
            for i, name in enumerate(self.coord_order)
        }

    def compute_normal(
        self, bound: dict, p: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the frozen outward unit normal at parameter values p.

        Uses autograd on the parametric curve (x(p), y(p)) — valid only
        for dim == 2.  For dim == 1 the notion of a normal is ill-defined;
        raise an error so callers handle it explicitly.

        The returned tensors have requires_grad=False (detached): normals
        are geometric constants, not learnable parameters.

        Args:
            bound: single boundary dict with keys "x", "y", "p", optionally "normal_dir"
            p:     (n,) parameter values in [p_min, p_max]

        Returns:
            nx: (n, 1)
            ny: (n, 1)
        """
        if len(p) == 0:
            return torch.zeros(0, 1), torch.zeros(0, 1)
        if self.dim != 2:
            raise RuntimeError("compute_normal is only defined for dim=2.")

        p_ = p.clone().detach().requires_grad_(True)
        x = bound["x"](p_)
        y = bound["y"](p_)

        ones = torch.ones_like(x)
        dx = torch.autograd.grad(x, p_, grad_outputs=ones, create_graph=False)[0]
        dy = torch.autograd.grad(y, p_, grad_outputs=ones, create_graph=False)[0]

        nx_raw = dy.detach()
        ny_raw = -dx.detach()
        norm = torch.sqrt(nx_raw**2 + ny_raw**2).clamp(min=1e-8)
        nx_raw = nx_raw / norm
        ny_raw = ny_raw / norm

        eps = 1e-3
        x0 = x.detach()
        y0 = y.detach()
        
        mid = len(p) // 2
        test_out = torch.stack([
            y0[mid:mid+1] + eps * ny_raw[mid:mid+1],
            x0[mid:mid+1] + eps * nx_raw[mid:mid+1],
        ], dim=1)

        is_inside = self.inside_spatial(test_out)
        sign = -1.0 if is_inside.item() else 1.0

        return (sign * nx_raw).unsqueeze(1), (sign * ny_raw).unsqueeze(1)

    def inside(self, coords: torch.Tensor, n_disc: int = 200) -> torch.Tensor:
        unpacked = self.unpack_coords(coords)
        if self.dim == 1:
            qx = unpacked["x"].squeeze(1)
            with torch.no_grad():
                x_vals = torch.cat([
                    torch.stack([bound["x"](torch.tensor([bound["p"][0], bound["p"][1]]))])
                    for bound in self.boundaries.values()
                ]).flatten()
            return (qx >= x_vals.min()) & (qx <= x_vals.max())
        return self._inside_core(unpacked["x"], unpacked["y"], n_disc)

    def inside_spatial(self, coords: torch.Tensor, n_disc: int = 200) -> torch.Tensor:
        if self.dim == 1:
            qx = coords[:, 0:1]

            with torch.no_grad():
                x_vals = []
                for bound in self.boundaries.values():
                    p0, p1 = bound["p"]
                    p_test = torch.tensor([p0, p1])
                    x_vals.append(bound["x"](p_test))

                x_vals = torch.cat(x_vals)

            return (qx >= x_vals.min()) & (qx <= x_vals.max())

        qy = coords[:, 0:1]
        qx = coords[:, 1:2]
        return self._inside_core(qx, qy, n_disc)

    def _inside_core(self, qx, qy, n_disc: int = 200) -> torch.Tensor:
        n_points = qx.shape[0]
        n_rays = 5
        votes = torch.zeros(n_points, dtype=torch.long)

        with torch.no_grad():
            for _ in range(n_rays):
                angle = torch.empty(1).uniform_(0.1, math.pi - 0.1).item()
                dx = math.cos(angle)
                dy = math.sin(angle)
                crossings = torch.zeros(n_points, dtype=torch.long)

                for bound in self.boundaries.values():
                    p_vals = torch.linspace(*bound["p"], n_disc)
                    x_vals = bound["x"](p_vals)
                    y_vals = bound["y"](p_vals)

                    ax = x_vals[:-1].unsqueeze(0)
                    ay = y_vals[:-1].unsqueeze(0)
                    bx = x_vals[1:].unsqueeze(0)
                    by = y_vals[1:].unsqueeze(0)

                    ex = bx - ax
                    ey = by - ay
                    fx = qx - ax
                    fy = qy - ay

                    denom = ex * dy - ey * dx
                    s = (fx * dy - fy * dx) / (denom + 1e-12)
                    t = (fx * ey - fy * ex) / (denom + 1e-12)

                    hit = (s >= 0) & (s <= 1) & (t > 0)
                    crossings += hit.sum(dim=1)

                votes += (crossings % 2 == 1).long()

        return votes > (n_rays // 2)
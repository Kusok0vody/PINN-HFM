import torch
import torch.nn as nn

from src.utils import unpack_coords_grad


class PINN(nn.Module):
    """
    Core PINN block. Connects net, physics and sampler.
    """

    def __init__(self, net, physics, sampler, device="cpu"):
        super().__init__()
        self.net     = net
        self.physics = physics
        self.sampler = sampler
        self.points  = None
        self.device  = torch.device(device)
        
        self.net.to(self.device)

        if hasattr(physics, 'param_order') and hasattr(net, 'mu_dim'):
            expected = len(physics.param_order)
            if net.mu_dim != expected:
                raise ValueError(
                    f"Net expects mu_dim={net.mu_dim}, "
                    f"but physics has {expected} parameters: {physics.param_order}"
                )

    def resample(self):
        """Generates a new pool of collocation points."""
        self.points = self.sampler.sample()
        self._points_to_device()

    def _points_to_device(self):
        """Moves all sampled point tensors to device."""
        pts = self.points
        pts.interior.coords = pts.interior.coords.to(self.device)

        for batch in pts.boundaries.values():
            batch.coords = batch.coords.to(self.device)
            batch.nx     = batch.nx.to(self.device)
            batch.ny     = batch.ny.to(self.device)

        if pts.initial is not None:
            pts.initial.coords = pts.initial.coords.to(self.device)

    def step(self, mu: torch.Tensor) -> dict:
        """
        Computes all residuals for a given parameter batch.

        Args:
            mu: (M, mu_dim) — parameter settings

        Returns:
            dict with keys "pde", "bc", "ic", "extra"
            each value is a dict of named residual tensors
        """
        if self.points is None:
            self.resample()

        M = mu.shape[0]

        # --- PDE ---
        coords_raw = self.points.interior.coords
        unpacked   = unpack_coords_grad(coords_raw, self.physics.has_time, self.physics.dim)
        coords_cat = torch.cat(list(unpacked.values()), dim=1)
        pred_pde   = self.physics.apply_transforms(self.net(coords_cat, mu))
        res_pde    = self.physics.residualPDE(pred_pde, unpacked, M)

        # --- BC ---
        res_bc = {}
        for name, batch in self.points.boundaries.items():
            coords_bc = batch.coords.requires_grad_(True)
            pred_bc   = self.physics.apply_transforms(self.net(coords_bc, mu))
            res_bc.update(self.physics.residualBC(pred_bc, batch, M))

        # --- IC ---
        res_ic = {}
        if self.points.initial is not None:
            coords_ic = self.points.initial.coords
            pred_ic   = self.physics.apply_transforms(self.net(coords_ic, mu))
            res_ic    = self.physics.residualIC(pred_ic, coords_ic, M)

        # --- Extra ---
        res_extra = self.physics.residualExtra(pred_pde, coords_cat, M)

        return {
            "pde":   res_pde,
            "bc":    res_bc,
            "ic":    res_ic,
            "extra": res_extra,
        }

    def __repr__(self) -> str:
        lines = ["PINN("]
        lines.append(f"  net:     {type(self.net).__name__}")
        lines.append(f"  physics: {type(self.physics).__name__}")
        lines.append(f"  sampler: {type(self.sampler).__name__}")
        lines.append(f"  device:  {self.device}")
        lines.append(")")
        return "\n".join(lines)
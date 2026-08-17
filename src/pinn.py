import math
import torch
import torch.nn as nn
from tqdm import tqdm

from utils import unpack_coords


class PINN(nn.Module):
    """
    Core PINN block. Connects net, physics and sampler.
    """

    def __init__(
        self, net,
        physics,
        sampler,
        n_refine: int      = 1,
        adaptive_pde: bool = True,
        adaptive_bc:  bool = False,
        adaptive_ic:  bool = False,
        autoscale_inputs: bool = True,
        paired_coords: bool = False,
        device="cpu"
        ):
        from geometry.sampler import Sampler, SampledPoints
        from physics.phys import Physics
        from network.net import Net
        
        super().__init__()
        self.net:Net               = net
        self.physics: Physics      = physics
        self.sampler: Sampler      = sampler
        self.points: SampledPoints = None
        self.n_refine              = n_refine
        self.device                = torch.device(device)
        
        self.adaptive_pde = adaptive_pde
        self.adaptive_bc  = adaptive_bc
        self.adaptive_ic  = adaptive_ic
        self.paired_coords = paired_coords
        
        self.net.to(self.device)

        if autoscale_inputs:
            self.autoscale_inputs()

    def autoscale_inputs(self):
        """
        Derive the network's input rescaling from the problem rather than from
        a hand-tuned constant.

        Coordinate bounds come from the sampler's bounding box, which is already
        aligned with Geometry.coord_order and therefore with the column order
        the network is fed. Parameter bounds come from physics.limits where a
        sweep is defined, and from the spread of the current batch otherwise;
        a single fixed parameter leaves that axis as the identity.
        """
        lows, highs = self.sampler._bounding_box()

        par        = self.physics.par
        mu_lo      = par.tensor.min(dim=0).values.clone()
        mu_hi      = par.tensor.max(dim=0).values.clone()
        mu_log     = torch.zeros(len(par.names), dtype=torch.bool)

        for i, name in enumerate(par.names):
            lim = getattr(self.physics, "limits", {}).get(name)
            if lim is None:
                continue
            lo, hi = float(lim["min"]), float(lim["max"])
            # Match the rescaling to how the sweep is actually drawn: a
            # log-uniform parameter mapped linearly would reach the network
            # with almost all of its mass bunched at one end.
            if lim.get("scale", "linear") == "log" and lo > 0.0:
                mu_log[i] = True
                lo, hi = math.log(lo), math.log(hi)
            mu_lo[i], mu_hi[i] = lo, hi

        self.net.set_input_bounds(
            x_lo=lows, x_hi=highs, mu_lo=mu_lo, mu_hi=mu_hi, mu_log=mu_log
        )

    def resample(self):
        """Generates a new pool of collocation points."""
        self.points = self.sampler.sample().to(self.device)

    def _safe_multinomial(self, weights: torch.Tensor, n: int) -> torch.Tensor:
        s = weights.sum()
        if s.item() < 1e-12:
            weights = torch.ones_like(weights)
            s = weights.sum()
        return torch.multinomial(weights / s, n, replacement=False)

    def resample_adaptive(self):
        from geometry.sampler import SampledPoints, BoundaryBatch, CollocationBatch

        for _ in (range(self.n_refine)):
            new_points = self.sampler.sample().to(self.device)

            if (self.adaptive_pde):
                pde_combined = torch.cat([
                    self.points.interior.coords,
                    new_points.interior.coords
                ])
                unpacked, pred_pde = self._evaluate(pde_combined, self.physics.par.tensor)
                res_pde = self.physics.residualPDE(pred_pde, unpacked)
                sumres_pde = torch.zeros(len(pde_combined), device=self.device)
                for _, res in res_pde.items():
                    sumres_pde += res.abs().mean(dim=1).detach()
                
                idx_pde    = self._safe_multinomial(sumres_pde, self.sampler.n_interior)
                interior   = pde_combined[idx_pde]
            else:
                interior = new_points.interior.coords
            
            if (self.adaptive_bc):
                boundaries = {}
                
                for name, batch in self.points.boundaries.items():
                    new_batch = new_points.boundaries[name]
                    n = self.sampler._n_boundary(name)

                    bc_combined = torch.cat([batch.coords, new_batch.coords])
                    nx_combined = torch.cat([batch.nx, new_batch.nx])
                    ny_combined = torch.cat([batch.ny, new_batch.ny])

                    combined_batch = BoundaryBatch(
                        coords=bc_combined,
                        nx=nx_combined,
                        ny=ny_combined,
                        name=name,
                    )

                    unpacked, pred_bc = self._evaluate(bc_combined, self.physics.par.tensor)
                    res_bc = self.physics.residualBC(pred_bc, unpacked, combined_batch)

                    sumres_bc = torch.zeros(len(bc_combined), device=self.device)
                    for _, res in res_bc.items():
                        sumres_bc += res.abs().mean(dim=1).detach()

                    idx_bc    = self._safe_multinomial(sumres_bc, n).tolist()
                    boundaries[name] = BoundaryBatch(
                        coords=bc_combined[idx_bc],
                        nx=nx_combined[idx_bc],
                        ny=ny_combined[idx_bc],
                        name=name,
                    )
            else:
                boundaries = new_points.boundaries
            
            initial = None
            if self.points.initial is not None:
                if (self.adaptive_ic):
                    ic_combined = torch.cat([
                        self.points.initial.coords,
                        new_points.initial.coords
                    ])
                    unpacked, coords_ic = unpack_coords(
                        ic_combined, self.physics.has_time, self.physics.dim
                    )
                    pred_ic   = self.physics.apply_transforms(self.net(coords_ic, self.physics.par.tensor))
                    pred_ic   = self.physics.apply_output_ansatz(pred_ic, unpacked)
                    res_ic    = self.physics.residualIC(pred_ic, coords_ic)
                    sumres_ic = torch.zeros(len(ic_combined), device=self.device)
                    for _, res in res_ic.items():
                        sumres_ic += res.abs().mean(dim=1).detach()
                    
                    idx_ic    = self._safe_multinomial(sumres_ic, self.sampler.n_initial).tolist()
                    initial   = ic_combined[idx_ic]
                else:
                    initial = new_points.initial.coords

            self.points = SampledPoints(
                interior=CollocationBatch(coords=interior),
                boundaries=boundaries,
                initial=CollocationBatch(coords=initial) if initial is not None else None,
            ).to(self.device)

    def _evaluate(self, coords: torch.Tensor, params: torch.Tensor):
        """
        Prediction on coordinates that will be differentiated, with transforms
        and the hard-constraint ansatz applied.

        With paired_coords the coordinates are replicated per parameter setting
        so each pair owns its leaf, which turns the M backward passes needed to
        separate the settings into one. Measured on CDR: x7.7 at four settings,
        x11.9 at eight, and the broadcast path runs out of memory at sixteen
        where this one does not.

        The flag is honoured whatever M is. At M = 1 the two layouts describe
        the same computation — one leaf per point either way — and
        derivative_batched then dispatches both to the same loop, which runs
        exactly once. That is worth stating because the alternative, silently
        ignoring the flag at M = 1, makes a run's configuration depend on how
        many parameter settings it happened to be given.

        Returns:
            (unpacked, pred) — coordinate leaves and the prediction, both keyed
            so that Physics sees the same shapes either way.
        """
        from utils import unpack_coords_paired

        if self.paired_coords:
            unpacked, X = unpack_coords_paired(
                coords, self.physics.has_time, self.physics.dim, params.shape[0]
            )
            pred = self.net.forward_paired(X, params)
        else:
            unpacked, coords_out = unpack_coords(
                coords, self.physics.has_time, self.physics.dim, requires_grad=True
            )
            pred = self.net(coords_out, params)

        pred = self.physics.apply_transforms(pred)
        return unpacked, self.physics.apply_output_ansatz(pred, unpacked)

    def predict(self, coords: torch.Tensor, params: torch.Tensor = None,
                requires_grad: bool = False) -> dict:
        """
        Forward pass with output transforms and the hard-constraint ansatz applied.

        This is what any consumer of a trained network should call: the raw
        net(...) output is pre-transform and pre-ansatz, and is therefore not
        the solution of the problem.

        Args:
            coords:        (N, n_coords)
            params:        (M, mu_dim); defaults to the current physics parameters
            requires_grad: keep the graph on the coordinate columns, so the
                           caller can differentiate the prediction

        Returns:
            dict: name -> (N, M)
        """
        params = self.physics.par.tensor if params is None else params

        unpacked, coords_out = unpack_coords(
            coords, self.physics.has_time, self.physics.dim, requires_grad=requires_grad
        )
        pred = self.physics.apply_transforms(self.net(coords_out, params))
        return self.physics.apply_output_ansatz(pred, unpacked)

    def step(self) -> dict:
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
            # self.resample_adaptive()

        parameters = self.physics.par.tensor

        # --- PDE ---
        pde_coords, pred_pde = self._evaluate(self.points.interior.coords, parameters)
        res_pde    = self.physics.residualPDE(pred_pde, pde_coords)

        # --- BC ---
        res_bc_raw = {}
        for _, batch in self.points.boundaries.items():
            unpacked, pred_bc = self._evaluate(batch.coords, parameters)
            res_bc_raw.update(self.physics.residualBC(pred_bc, unpacked, batch))
        res_bc = self.physics.apply_boundary_constraints(res_bc_raw)

        # --- IC ---
        res_ic = {}
        if self.points.initial is not None:
            unpacked, coords_ic = unpack_coords(
                self.points.initial.coords, self.physics.has_time, self.physics.dim
            )
            pred_ic   = self.physics.apply_transforms(self.net(coords_ic, parameters))
            pred_ic   = self.physics.apply_output_ansatz(pred_ic, unpacked)
            res_ic    = self.physics.residualIC(pred_ic, coords_ic)

        # --- Extra ---
        res_extra = self.physics.residualExtra(pred_pde, pde_coords)

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
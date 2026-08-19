import math
import pathlib

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
        scale_free_pde: bool = False,
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
        self.scale_free_pde = scale_free_pde
        self.last_scale     = None
        self.data           = None
        self._data_source   = None
        self._data_follows  = False
        self._data_resample = False
        self.use_data       = True

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
                # Same units as the training loss. The point pool is drawn by
                # comparing residuals across settings as well as across points,
                # so leaving these raw while the loss is scale-free would hand
                # nearly every collocation point to whichever setting has the
                # largest solution.
                res_pde = self._descale(
                    self.physics.residualPDE(pred_pde, unpacked),
                    scale=self._field_scale(pred_pde),
                )
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

        Coordinates are replicated per parameter setting so each pair owns its
        leaf, which turns the M backward passes needed to separate the settings
        into one. Measured on CDR: x7.7 at four settings, x11.9 at eight, and
        the shared-leaf path runs out of memory at sixteen where this one does
        not. The two agree bitwise on the residual and to 1e-6 on the gradients
        that reach the weights, so there is nothing to choose between them on
        correctness and no reason to keep the choice.

        At M = 1 the two layouts describe the same computation — one leaf per
        point either way — and derivative_batched dispatches both to the same
        loop, which runs exactly once.

        Returns:
            (unpacked, pred) — coordinate leaves and the prediction, both keyed
            so that Physics sees the same shapes either way.
        """
        from utils import unpack_coords_paired

        unpacked, X = unpack_coords_paired(
            coords, self.physics.has_time, self.physics.dim, params.shape[0]
        )
        pred = self.net.forward_paired(X, params)

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

    # A field of amplitude 1e-8 is a network that has not started rather than a
    # solution, and dividing by its own size would turn numerical dust into a
    # full-sized residual.
    SCALE_FLOOR = 1e-6

    def _field_scale(self, pred: dict, detach: bool = True) -> torch.Tensor:
        """
        Per-setting size of the predicted solution, (M,).

        The divisor keeps its graph when it is used to descale, and that is not
        a detail. Detached, the ratio is invariant in value but not in
        gradient: d/dg (r / s_det) = r / s, a push towards a smaller field that
        does not weaken as the field shrinks. The boundary term, the only thing
        pulling the amplitude back up, has a gradient proportional to the
        output itself and dies out as the field vanishes. A run with that
        combination walks into u = 0 and stops — measured, not feared: 30000
        steps ending at exactly zero, residual 2e-9, boundary error exactly 1.

        Attached, the ratio is homogeneous of degree zero in the field, so the
        equation term has no opinion about amplitude at all and the boundary
        data set it alone. The cost is that the ratio can also be reduced by
        inflating the field with something the equation does not charge for —
        a near-eigenmode — which the boundary term then has to hold in check.
        """
        sq = torch.stack([(v.detach() if detach else v).pow(2).mean(dim=0)
                          for v in pred.values()])
        return sq.mean(dim=0).sqrt().clamp_min(self.SCALE_FLOOR)

    def _descale(self, residuals: dict, scale: torch.Tensor = None) -> dict:
        """
        Divide interior residuals by the size of the field that produced them.

        The equation's residual carries the dimension of the solution: for a
        field of amplitude A approximated to relative accuracy d, it is of size
        d*A, while the boundary error stays of size (1 - alpha) whatever A is.
        Adding those two directly makes the best trade-off between them depend
        on A, and for large A the cheapest answer is a damped field: the PDE is
        linear, so alpha*u solves it exactly for any alpha, and only the
        boundary term objects. Dividing here removes A from that comparison, so
        the optimum in alpha sits at 1 regardless of how large the solution is.
        See _field_scale for why the divisor keeps its graph.

        Boundary and initial residuals are deliberately left alone. Their
        natural scale is the data they are matched against, which is already
        the right unit; dividing them by the field size would say that a
        boundary error is acceptable in proportion to how large the interior
        happens to be.
        """
        scale = self.last_scale if scale is None else scale
        if not self.scale_free_pde or scale is None:
            return residuals
        return {k: v / scale for k, v in residuals.items()}

    def _draw_data_points(self, n_points: int, seed: int = None):
        """
        N_data points from the sampler, uniformly over the domain.

        Uniform on purpose. Placing them where the residual is currently large
        is tempting and wrong for the same reason the collocation pool was the
        wrong home for them: measurements do not chase the network's present
        error, and letting them do so ties two mechanisms that have to stay
        independent to be read separately.

        One sample() yields n_interior points, which may be fewer than asked
        for, so this draws until there are enough rather than silently
        installing a smaller set.
        """
        g = torch.random.get_rng_state()
        if seed is not None:
            torch.manual_seed(seed)
        drawn, have = [], 0
        while have < n_points:
            c = self.sampler.sample().interior.coords
            drawn.append(c)
            have += c.shape[0]
        coords = torch.cat(drawn)[:n_points]
        torch.random.set_rng_state(g)
        return coords

    def set_data(self, source, n_points: int = 256, coords=None, params=None,
                 seed: int = 0, resample_points: bool = False):
        """
        Install reference or measured values as an extra loss term.

        N_data points, fixed in space for the whole run, with their values held
        in memory. The loss term is then simply net(x_data) - data over all of
        them.

        A problem has one source of such values, chosen by what it costs to
        produce them. They are never mixed.

            callable   fn(coords, params) -> {name: (N, M)}. The values follow
                       the swept parameters: whenever the parameter batch is
                       resampled the source is consulted again, at the same
                       points. That is what makes the term mean anything during
                       a sweep — values at settings the network is not currently
                       training on say nothing about the ones it is.
            path       .npz with "coords", "params" and one array per variable.
                       Fixed by construction: a file cannot answer at settings
                       nobody computed, so these stay put while the sweep moves.
            dict       {name: (N, M)} in memory, with coords and params given.

        Where a problem could offer either, a callable wins: it is exact at
        whatever points are asked of it, so the data can match the geometry and
        the sweep in use, while a file is a snapshot of an earlier computation
        on its own points at its own settings.

        Why data belong here at all: the residual does not select a solution
        where the problem is degenerate. A homogeneous equation admits any
        multiple of its solution, a parameter near an eigenvalue admits any
        admixture of the eigenmode, and the objective scores all of them alike.
        A handful of values does what no weighting can — it names which member
        is meant.

        Args:
            source:   callable, path, or dict of arrays as above
            n_points: N_data, drawn from the sampler when coords are not given
            coords:   (N, n_coords) in Geometry order, to place them by hand
            params:   (M, mu_dim), only for a dict source; a callable follows
                      the sweep and a file carries its own
            seed:     fixes the draw, so the points are part of the config
            resample_points:
                      redraw the points whenever the values are recomputed, i.e.
                      once per parameter resample, not per iteration. Costs
                      nothing, since the source is being consulted at that moment
                      anyway, and turns the term from a fixed set of measurements
                      into a Monte Carlo estimate of the continuous error.

                      Off by default, and the default is the claim: a fixed set
                      is what a measurement is, and it is the only form available
                      to a problem whose reference came from a file, so it is
                      what keeps problems comparable. Switching it on makes the
                      analytic problems an oracle answering anywhere.

                      The reason to have the switch at all is that the difference
                      between the two settings measures something: if they agree,
                      the network is not memorising N_data points and the count is
                      sufficient; if they disagree, it is too small. Note that
                      even with fixed points the targets are not fixed — they are
                      recomputed at every new parameter batch — so what could be
                      memorised is already a function of mu at those points
                      rather than a table.
        """
        if source is None:
            if not self.physics.has_reference:
                raise ValueError(
                    f"{type(self.physics).__name__} has no reference, so a data "
                    f"source must be given explicitly"
                )
            source = self.physics.reference

        if isinstance(source, (list, tuple)):
            candidates = [c for c in source if c is not None]
            chosen = next((c for c in candidates if callable(c)), None)
            if chosen is None:
                chosen = next((c for c in candidates
                               if isinstance(c, (str, pathlib.Path))
                               and pathlib.Path(c).exists()), None)
            if chosen is None:
                raise ValueError(
                    f"no usable data source among {source}: no callable, and no "
                    f"file that exists"
                )
            if len(candidates) > 1:
                print(f"PINN: {len(candidates)} data sources offered, using the "
                      f"{'function' if callable(chosen) else 'file'}")
            source = chosen

        if isinstance(source, (str, pathlib.Path)):
            import numpy as np
            blob   = np.load(str(source))
            coords = torch.as_tensor(blob["coords"], dtype=torch.float32)
            params = torch.as_tensor(blob["params"], dtype=torch.float32)
            values = {k: torch.as_tensor(blob[k], dtype=torch.float32)
                      for k in blob.files if k not in ("coords", "params")}
            follows = False
        elif callable(source):
            if coords is None:
                coords = self._draw_data_points(n_points, seed)
            params  = self.physics.par.tensor.detach().cpu()
            values  = source(coords, params)
            follows = True
        else:
            if coords is None or params is None:
                raise ValueError(
                    "coords and params must both be given when source is an array"
                )
            values  = dict(source)
            follows = False

        self._data_source  = source
        self._data_follows = follows
        self._data_resample = resample_points and follows
        if resample_points and not follows:
            print("PINN: resample_points has no effect on data read from a file; "
                  "its points are whatever was computed in advance")
        self.data = {"coords": coords.to(self.device), "params": None, "values": {}}
        self._install_values(params, values)

        n, m = self.data["coords"].shape[0], self.data["params"].shape[0]
        where = "redrawn each resample" if self._data_resample else "fixed"
        print(f"PINN: {n} data points ({where}), {m} parameter settings"
              f"{' following the sweep' if follows else ' fixed'}, "
              f"variables {sorted(self.data['values'])}")

    def _install_values(self, params, values):
        """Check a set of values against the installed points, then store it."""
        params = torch.as_tensor(params, dtype=torch.float32).reshape(
            -1, self.physics.par.tensor.shape[1]
        )
        n, m = self.data["coords"].shape[0], params.shape[0]

        checked = {}
        for name, v in values.items():
            v = torch.as_tensor(v, dtype=torch.float32)
            if v.dim() == 1:
                v = v.unsqueeze(1)
            if v.shape != (n, m):
                raise ValueError(
                    f"data['{name}'] has shape {tuple(v.shape)}, expected "
                    f"{(n, m)} from {n} points and {m} parameter settings"
                )
            if not torch.isfinite(v).all():
                raise ValueError(
                    f"data['{name}'] contains non-finite values; an analytic "
                    f"reference returns NaN outside its domain, so check that "
                    f"the points are inside it"
                )
            checked[name] = v.to(self.device)

        unknown = set(checked) - set(self.physics.transforms)
        if unknown:
            raise ValueError(
                f"data names {sorted(unknown)} are not outputs of this physics "
                f"({sorted(self.physics.transforms)})"
            )
        self.data["params"] = params.to(self.device)
        self.data["values"] = checked

    def update_data(self):
        """
        Recompute the values at the current parameter settings.

        Called when the sweep draws a new parameter batch. Only a callable
        source can answer at new settings; a file stays where it is, which is
        not a limitation to work around but the honest state of a problem whose
        reference had to be computed in advance.

        The cost is the source's, on N_data points. For the annulus, 256 points
        at 32 settings take about a second, against a second of training per
        hundred steps, so recomputing at every parameter resample is affordable.
        Recomputing on the collocation pool would not be: it is an order larger
        and turns over ten times as often, which measured out at sixteen seconds
        a time and would have doubled the length of a run.
        """
        if self.data is None or not self._data_follows:
            return
        if self._data_resample:
            n = self.data["coords"].shape[0]
            self.data["coords"] = self._draw_data_points(n).to(self.device)
        params = self.physics.par.tensor.detach().cpu()
        values = self._data_source(self.data["coords"].cpu(), params)
        self._install_values(params, values)

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
        # Attached for the division, detached for reporting.
        scale           = self._field_scale(pred_pde, detach=not self.scale_free_pde)
        self.last_scale = scale.detach()
        res_pde    = self._descale(res_pde, scale)

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
        res_extra = self._descale(self.physics.residualExtra(pred_pde, pde_coords))

        # --- Data ---
        # Not descaled, for the same reason the boundary terms are not: the
        # natural unit of a measurement is the measurement.
        res_data = {}
        if self.data is not None and self.use_data:
            pred_data = self.predict(self.data["coords"], self.data["params"])
            res_data = {name: pred_data[name] - target
                        for name, target in self.data["values"].items()}

        return {
            "pde":   res_pde,
            "bc":    res_bc,
            "ic":    res_ic,
            "extra": res_extra,
            "data":  res_data,
        }
        
    def __repr__(self) -> str:
        lines = ["PINN("]
        lines.append(f"  net:     {type(self.net).__name__}")
        lines.append(f"  physics: {type(self.physics).__name__}")
        lines.append(f"  sampler: {type(self.sampler).__name__}")
        lines.append(f"  device:  {self.device}")
        lines.append(")")
        return "\n".join(lines)
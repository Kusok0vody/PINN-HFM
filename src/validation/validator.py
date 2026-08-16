import torch

from physics.parameters import ParamBatch
from validation.metrics import relative_l2, residual_norms


class Validator:
    """
    Out-of-sample quality monitor for a trained or training PINN.

    Provides two independent signals:

      holdout residual
          PDE/BC/IC residuals evaluated on a pool of points the optimiser never
          sees. The training loss is a biased estimate of quality whenever
          adaptive resampling is on, because resampling deliberately
          concentrates points where the residual is already large. The holdout
          pool is drawn once at construction and never refreshed, so successive
          evaluations stay comparable. Works for any problem — no reference
          solution needed.

      reference error
          Relative L2 against a precomputed reference (analytic or from a
          numerical solver). Optional: several problems have no cheap reference.

    The parameter batch is frozen as well. With Trainer(param_every > 0) the
    physics parameters are resampled during training, which would otherwise
    make successive validations incomparable and the reference meaningless.

    Args:
        pinn:       PINN instance
        ref_coords: (N, n_coords) points at which the reference is known
        ref_values: {var_name: (N, M_ref)} reference values; numpy is accepted
        ref_params: (M_ref, mu_dim) parameters the reference corresponds to.
                    Defaults to the parameters currently held by physics.
        holdout:    draw a holdout pool (set False to only track the reference)
        seed:       seed for drawing the holdout pool, for reproducible runs
    """

    def __init__(
        self,
        pinn,
        ref_coords: torch.Tensor = None,
        ref_values: dict         = None,
        ref_params: torch.Tensor = None,
        holdout:    bool         = True,
        seed:       int          = None,
    ):
        self.pinn   = pinn
        self.device = pinn.device

        par_tensor = pinn.physics.par.tensor if ref_params is None else ref_params
        self.par   = ParamBatch(
            par_tensor.detach().clone().to(self.device),
            list(pinn.physics.par.names),
        )

        if holdout:
            if seed is None:
                self.holdout = pinn.sampler.sample().to(self.device)
            else:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed)
                    self.holdout = pinn.sampler.sample().to(self.device)
        else:
            self.holdout = None

        self.ref_coords = (
            ref_coords.to(self.device) if ref_coords is not None else None
        )
        self.ref_values = None
        if ref_values is not None:
            self.ref_values = {
                name: torch.as_tensor(val).detach().to(self.device)
                for name, val in ref_values.items()
            }

        if (self.ref_coords is None) != (self.ref_values is None):
            raise ValueError(
                "ref_coords and ref_values must be provided together."
            )

    def _swap_state(self):
        """Temporarily install the frozen holdout pool and parameter batch."""
        return (self.pinn.points, self.pinn.physics.par)

    def holdout_residual(self) -> dict:
        """
        Residual norms on the frozen holdout pool.

        Returns:
            {"holdout/<group>/<name>": mse, ..., "holdout/total": mse}
        """
        if self.holdout is None:
            return {}

        saved_points, saved_par = self._swap_state()
        try:
            self.pinn.points      = self.holdout
            self.pinn.physics.par = self.par
            norms = residual_norms(self.pinn.step())
        finally:
            self.pinn.points      = saved_points
            self.pinn.physics.par = saved_par

        return {f"holdout/{key}": val for key, val in norms.items()}

    def reference_error(self) -> dict:
        """
        Relative L2 against the reference solution, one entry per variable.

        With M_ref > 1 parameter settings the error is computed over the whole
        block at once, so a sweep collapses to a single number per variable.

        Returns:
            {"l2/<var>": error}
        """
        if self.ref_values is None:
            return {}

        saved_par = self.pinn.physics.par
        try:
            self.pinn.physics.par = self.par
            with torch.no_grad():
                pred = self.pinn.predict(self.ref_coords, self.par.tensor)
        finally:
            self.pinn.physics.par = saved_par

        return {
            f"l2/{name}": relative_l2(pred[name], ref)
            for name, ref in self.ref_values.items()
        }

    def evaluate(self) -> dict:
        """All available metrics in one flat dict, ready for logging."""
        metrics = {}
        metrics.update(self.holdout_residual())
        metrics.update(self.reference_error())
        return metrics

import os
import time
import torch
from tqdm import tqdm
import datetime


class Trainer:
    """
    Training loop for PINN.

    Responsibilities:
        - optimiser and lr-scheduler
        - loss aggregation from residuals with GradNorm-balanced weights
        - logging via tqdm + tensorboard
        - checkpointing (net state_dict + optimiser + sampled points)
        - calling resample() every resample_every iterations
        - out-of-sample validation via an optional Validator

    Args:
        pinn:             PINN instance
        lr:               initial learning rate
        n_iter:           number of training iterations
        resample_every:   resample collocation points every K iterations
        checkpoint_every: save checkpoint every K iterations
        checkpoint_path:  directory for checkpoints
        logger:           "tqdm" or "tensorboard"
        device:           "cpu" or "cuda"
        gradnorm_every:   rebalance loss weights every K iterations
        lra_alpha:        weight given to the freshly measured ratio when
                          updating a balancing weight, so small means heavy
                          smoothing. Kept in this direction because six
                          experiment scripts already pass 0.01 and flipping
                          the sense silently turns strong smoothing into none
        weight_min,
        weight_max:       clamp on the balancing weights; the LRA ratio is
                          unbounded and diverges without it
        param_every:      resample physics parameters every K iterations (0 = never)
        use_data:         include the data term when data are installed
        validator:        Validator instance, or None to skip validation
        validate_every:   run the validator every K iterations (0 = never)
        optimiser:        "nadam" or "hypergrad" (adapts its own learning rate)
        progress:         show the tqdm bar; turn off for non-interactive runs
        log_every:        print one compact status line every K iterations.
                          Intended for cluster logs, where a progress bar is
                          megabytes of noise but total silence for hours is
                          worse — there is no way to tell a slow run from a
                          hung one.
    """

    def __init__(
        self,
        pinn,
        lr:               float = 1e-3,
        n_iter:           int   = 10000,
        resample_every:   int   = 1000,
        checkpoint_every: int   = 1000,
        checkpoint_path:  str   = "checkpoints",
        logger:           str   = "tqdm",
        device:           str   = "cpu",
        run_name:         str   = None,
        save_final:       bool  = True,
        start_step:       int   = 0,
        gradnorm_every:   int   = 200,
        lra_alpha:        float = 0.01,
        param_every:      int   = 0,
        use_data:         bool  = True,
        validator               = None,
        validate_every:   int   = 0,
        weight_max:       float = 100.0,
        weight_min:       float = 0.01,
        optimiser:        str   = "nadam",
        progress:         bool  = True,
        log_every:        int   = 0,
        scheduler:        str   = "plateau",
        sched_patience:   int   = 2000,
        sched_factor:     float = 0.5,
        min_lr:           float = 1e-6,
    ):
        self.run_name        = run_name or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_final      = save_final
        self.checkpoint_path = os.path.join(checkpoint_path, self.run_name)
        
        self.pinn             = pinn
        self.n_iter           = n_iter
        self.resample_every   = resample_every
        self.checkpoint_every = checkpoint_every
        self.device           = device
        self.logger_type      = logger
        self.progress         = progress
        self.log_every        = log_every
        self.start_step       = start_step

        self.gradnorm_every   = gradnorm_every
        self.lra_alpha        = lra_alpha
        self.weight_max       = weight_max
        self.weight_min       = weight_min

        self.adaptive_weights = {}
        self._weights_initialized = False
        self._weights_saturated = False
        self.param_every = param_every

        # Data are installed on the PINN and switched on here, so that an
        # ablation is one flag in the training script rather than an edit to
        # where the data are built.
        pinn.use_data = use_data
        if use_data and pinn.data is None:
            print("Trainer: use_data is on but no data are installed; call "
                  "pinn.set_data(...) or the term is simply absent.")
        elif not use_data and pinn.data is not None:
            n = pinn.data["coords"].shape[0]
            print(f"Trainer: {n} data points are installed but use_data is off, "
                  f"so they take no part in training.")

        # Parameter sweeping needs two things that are set in different places:
        # limits on the Physics and a non-zero period here. Either one alone is
        # silently inert — draw_parameters returns nothing without limits, and
        # nothing calls it at param_every = 0 — so a run can look like a sweep
        # in the script, print no complaint, and train on one fixed batch for
        # its whole length. Say so instead.
        limits = getattr(pinn.physics, "limits", {}) or {}
        if self.param_every > 0 and not limits:
            print("Trainer: param_every > 0 but the physics has no limits, so "
                  "parameters will never be resampled. Pass limits=... to "
                  "setParameters to define the sweep.")
        elif self.param_every == 0 and limits:
            print(f"Trainer: limits are defined for {sorted(limits)} but "
                  f"param_every = 0, so the sweep is never drawn and training "
                  f"stays on the {pinn.physics.par.tensor.shape[0]} settings "
                  f"passed to setParameters.")
        elif not limits and pinn.physics.par.tensor.shape[0] > 1:
            # Legitimate on its own — a fixed set of settings is a valid run —
            # but it is also what a script looks like when it computed limits
            # and forgot to pass them, which is invisible from here: the physics
            # simply has no sweep. Stating the fact costs one line and is the
            # difference between "trained on a family" and "trained on ten
            # points", which are read from the same plots.
            print(f"Trainer: training on a fixed set of "
                  f"{pinn.physics.par.tensor.shape[0]} parameter settings, with "
                  f"no sweep. Pass limits=... to setParameters and param_every "
                  f"to resample them.")

        self.validator      = validator
        self.validate_every = validate_every
        self.last_metrics   = {}

        if optimiser == "nadam":
            self.optimiser = torch.optim.NAdam(pinn.net.parameters(), lr=lr)
        elif optimiser == "hypergrad":
            from training.adaptive import HyperGradAdam
            self.optimiser = HyperGradAdam(pinn.net.parameters(), lr=lr)
        else:
            raise ValueError(
                f"unknown optimiser '{optimiser}'; choose 'nadam' or 'hypergrad'"
            )
        self.optimiser_kind = optimiser

        # The plateau scheduler watches the training loss, which here jumps
        # whenever the collocation pool is redrawn, the loss weights are
        # rebalanced or the physics parameters are resampled. Measured on CDR
        # the median step-to-step change is ~3 percent, so a short patience
        # detects that noise rather than a real plateau: at patience=1000 the
        # rate fell from 1e-3 to min_lr by step 12000 of 20000, and the rest of
        # the run did nothing. Hence the longer default, and the option to turn
        # scheduling off entirely.
        if scheduler == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimiser, patience=sched_patience,
                factor=sched_factor, min_lr=min_lr,
            )
        elif scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimiser, T_max=max(n_iter, 1), eta_min=min_lr,
            )
        elif scheduler == "none":
            self.scheduler = None
        else:
            raise ValueError(
                f"unknown scheduler '{scheduler}'; choose 'plateau', 'cosine' or 'none'"
            )
        self.scheduler_kind = scheduler

        self._init_logger()

    def _init_logger(self):
        if self.logger_type == "tensorboard":
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter()
            except ImportError:
                print("tensorboard not installed: pip install tensorboard")
                self.logger_type = "tqdm"
                self.writer = None
        else:
            self.writer = None

    @staticmethod
    def _flat_grad(loss, params, retain: bool = True) -> torch.Tensor:
        grads = torch.autograd.grad(loss, params, retain_graph=retain, allow_unused=True)
        return torch.cat([
            (g if g is not None else torch.zeros_like(p)).flatten()
            for g, p in zip(grads, params)
        ])

    def _term_losses(self, residuals: dict) -> dict:
        """Unweighted per-term losses, keyed "group/name"."""
        return {
            f"{group}/{name}": (res ** 2).mean()
            for group, res_dict in residuals.items()
            for name, res in res_dict.items()
        }

    # Warn when the lightest and heaviest term differ by this factor. Chosen
    # from measurement, not taste. On the annulus with sixteen boundary arcs as
    # separate terms, 200 steps of LRA spanned 2.9x and the same 200 steps of
    # NTK weighting spanned 495x, on its way to the 1e4 the clamps allow; a
    # longer LRA run on the same problem spanned about 30x. The threshold sits
    # an order above healthy and well below the clamp, so it fires while the
    # objective still has time to be fixed rather than after the lightest terms
    # have stopped constraining anything.
    WEIGHT_SPREAD_WARN = 300.0

    def _check_weights(self, step: int):
        """
        Say something when the balancing weights pile up against the clamp.

        Every rule of the form lambda_i = mean_j(stat_j) / stat_i is a positive
        feedback loop for any term whose statistic collapses: satisfy a
        condition, its statistic shrinks, its weight grows, it is enforced
        harder still. The clamp stops the number but not the consequence — with
        most of the budget on a handful of terms the rest of the objective stops
        constraining anything.

        Measured here on the annulus with sixteen boundary arcs as separate
        terms: under NTK weighting the eight homogeneous inner-arc terms reached
        the clamp within a few dozen updates, the equation term was left with a
        negligible weight, and a smooth spurious field of rms 1.7 grew over the
        following fifteen thousand steps while the projection onto the exact
        solution stayed near one. Nothing in the loss curve says that is
        happening.

        The published rules are formulated for two to four loss terms. Splitting
        a boundary into arcs multiplies the count, and the mean in the numerator
        is then set by whichever terms happen to be large.
        """
        if len(self.adaptive_weights) < 2 or self._weights_saturated:
            return
        items = sorted(self.adaptive_weights.items(), key=lambda kv: kv[1])
        lo_k, lo = items[0]
        hi_k, hi = items[-1]
        if lo <= 0.0 or hi / lo < self.WEIGHT_SPREAD_WARN:
            return

        self._weights_saturated = True
        vals = [v for _, v in items]
        print("")
        print(f"Trainer: at step {step} the balancing weights span "
              f"{hi / lo:.3g}x — lowest {lo_k}={lo:.3g}, highest {hi_k}={hi:.3g}, "
              f"median {vals[len(vals) // 2]:.3g} over {len(vals)} terms. The "
              f"lowest-weighted terms are no longer constraining the solution. "
              f"For reference a healthy run on this problem spanned about 30x.",
              flush=True)

    def _update_weights_lra(self, residuals: dict):
        """
        Gradient-statistics balancing, a bounded variant of Learning Rate
        Annealing (Wang et al.):

            lambda_hat_i = mean_j ||grad_theta L_j|| / ||grad_theta L_i||
            lambda_i(t)  = (1 - alpha) lambda_i(t-1) + alpha clamp(lambda_hat_i)

        Every term is compared on the same statistic — the L2 norm of its
        gradient — and pulled towards the average of them all.

        This deviates from Eq. (6) of Bischof & Kraus, which divides the *max*
        entry of the governing term's gradient by the *mean* entry of term i's.
        That form was tried and reverted. Two reasons, both measured on the
        Helmholtz annulus: the max/mean mismatch inflates every weight by the
        ratio between those statistics, here 38x; and the paper's examples carry
        two to four loss terms while this problem carries seventeen, so the
        boundary side ends up outweighing the equation by four orders of
        magnitude. Weights reached a median of 3.6e3 against 6.2 here, training
        diverged, and the holdout residual climbed two hundredfold. The paper
        itself names the unboundedness as LRA's weakness and reports a problem
        it could not train at all.

        Despite the name this schedules the loss weights, not the learning rate.
        """
        params = [p for p in self.pinn.net.parameters() if p.requires_grad]
        losses = self._term_losses(residuals)
        if not losses:
            return

        norms = {key: self._flat_grad(loss, params).norm().item()
                 for key, loss in losses.items()}
        mean_norm = sum(norms.values()) / len(norms)

        for key, norm in norms.items():
            if norm < 1e-12:
                continue
            lam_hat = min(max(mean_norm / (norm + 1e-8), self.weight_min),
                          self.weight_max)
            old = self.adaptive_weights.get(key, 1.0)
            self.adaptive_weights[key] = (
                (1.0 - self.lra_alpha) * old + self.lra_alpha * lam_hat
            )

        self._check_weights(getattr(self, "_step", 0))
        self.optimiser.zero_grad()

    def _aggregate_loss(self, residuals: dict) -> tuple[torch.Tensor, dict]:
        loss_terms = {}
        total      = torch.tensor(0.0, device=self.device)

        for group, res_dict in residuals.items():
            for name, res in res_dict.items():
                key = f"{group}/{name}"
                w   = self.adaptive_weights.get(key, 1.0)
                term = w * (res**2).mean()
                loss_terms[key] = term.item()
                total = total + term

        return total, loss_terms

    # A field this much smaller than the one training started with is not a
    # solution in progress. The threshold is loose on purpose: legitimate early
    # transients move the amplitude by a factor of a few, not by a thousand.
    COLLAPSE_FACTOR = 1e-3

    def _check_collapse(self, step: int):
        """
        Say something when the solution is on its way to zero.

        u = 0 satisfies any linear homogeneous equation exactly, so a run that
        has collapsed reports a beautiful residual and a loss that stops moving,
        and looks from the outside like a run that converged. Worse, with a
        multiplicative output scale every gradient is proportional to the output
        itself, so the gradients vanish along with the field and the run cannot
        climb back out on its own. There is nothing to do but notice.
        """
        amp = float(self.pinn.last_scale.max())
        if getattr(self, "_amp0", None) is None:
            self._amp0 = amp
            self._collapse_reported = False
            return
        if (not self._collapse_reported
                and self._amp0 > 0.0 and amp < self.COLLAPSE_FACTOR * self._amp0):
            self._collapse_reported = True
            print("")
            print(f"Trainer: at step {step} the solution has shrunk to "
                  f"{amp:.2e}, {self._amp0 / max(amp, 1e-30):.0f}x smaller than "
                  f"at the start. u = 0 solves a linear homogeneous equation "
                  f"exactly and is a fixed point of a multiplicative output "
                  f"scale; the residual will look excellent from here on.",
                  flush=True)

    def _log(self, step: int, loss_terms: dict, total: float):
        if self.logger_type == "tensorboard":
            self.writer.add_scalar("loss/total", total, step)
            for name, val in loss_terms.items():
                self.writer.add_scalar(f"loss/{name}", val, step)

    def _validate(self, step: int) -> dict:
        """
        Runs the validator and logs its metrics under "val/".

        Unlike the training loss these are out-of-sample: the holdout pool and
        the parameter batch are frozen inside the Validator, so the numbers are
        comparable across steps and across runs.
        """
        metrics = self.validator.evaluate()
        self.last_metrics = metrics

        if self.logger_type == "tensorboard" and self.writer is not None:
            for name, val in metrics.items():
                self.writer.add_scalar(f"val/{name}", val, step)

        return metrics

    def _save_checkpoint(self, step: int):
        os.makedirs(self.checkpoint_path, exist_ok=True)
        torch.save(
            {
                "step":       step,
                "run_name":   self.run_name,
                "net":        self.pinn.net.state_dict(),
                "net_config": self.pinn.net.serialize_config(),
                "optimiser":  self.optimiser.state_dict(),
                # The balancing weights are state too, and they are the first
                # suspect whenever a run diverges: a term whose statistic
                # collapses takes its weight to the clamp and drags the whole
                # objective with it. Without them in the checkpoint that has to
                # be reconstructed by guesswork after the fact.
                "weights":    dict(self.adaptive_weights),
                # scheduler="none" leaves nothing to save, and step 0 is always
                # a checkpoint step, so without this the option crashes on its
                # first use rather than at some later moment.
                "scheduler":  (self.scheduler.state_dict()
                               if self.scheduler is not None else None),
                "points":     self.pinn.points,
            },
            f"{self.checkpoint_path}/ckpt_{step}.pt",
        )

    @staticmethod
    def load_checkpoint(path: str, pinn=None, optimiser=None, 
                        scheduler=None, device="cpu") -> tuple:
        """
        Loads checkpoint and restores training state. \\
        If pinn=None - get Net from checkpoint automaticaly. \\
        else - load weights

        Returns:
            (net, step)
        """
        from network.net import Net
        
        ckpt = torch.load(path, map_location=device, weights_only=False)

        if pinn is None:
            net = Net.from_checkpoint(path, device=device)
            net.to(device)
        else:
            Net.load_weights(pinn.net, ckpt["net"])
            pinn.net.to(device)
            net = pinn.net

        if optimiser is not None:
            optimiser.load_state_dict(ckpt["optimiser"])
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if pinn is not None and ckpt.get("points") is not None:
            pinn.points = ckpt["points"]

        return net, ckpt["step"]

    def train(self):
        self._t_start = time.time()
        self.pinn.resample()

        pbar = tqdm(range(self.start_step, self.start_step+self.n_iter+1),
                    desc="Training", disable=not self.progress)

        for step in pbar:
            self._step = step
            
            if self.param_every > 0 and step % self.param_every == 0:
                self.pinn.physics._resample_parameters()
                # Values that follow the sweep are recomputed here,
                # at the new settings and on the same points.
                self.pinn.update_data()
                # The objective just changed; an adapted step size carried
                # across that boundary refers to the previous problem.
                if hasattr(self.optimiser, "reset_lr"):
                    self.optimiser.reset_lr()
            
            if step > self.start_step and step % self.resample_every == 0:
                self.pinn.resample_adaptive()


            self.optimiser.zero_grad()

            residuals = self.pinn.step()

            if not self._weights_initialized or step % self.gradnorm_every == 0:
                self._update_weights_lra(residuals)
                self._weights_initialized = True
                residuals = self.pinn.step()

            total, loss_terms = self._aggregate_loss(residuals)

            total.backward()
            self.optimiser.step()
            if self.scheduler is not None:
                # cosine steps on the iteration count, plateau on the metric
                if self.scheduler_kind == "plateau":
                    self.scheduler.step(total.item())
                else:
                    self.scheduler.step()

            self._log(step, loss_terms, total.item())

            if (self.validator is not None and self.validate_every > 0
                    and step % self.validate_every == 0):
                self._validate(step)

            if self.log_every > 0 and step % self.log_every == 0:
                done = step - self.start_step + 1
                rate = done / max(time.time() - self._t_start, 1e-9)
                left = (self.n_iter + 1 - done) / max(rate, 1e-9)
                bits = [f"step {step}/{self.start_step + self.n_iter}",
                        f"loss {total.item():.3e}",
                        f"lr {self.optimiser.param_groups[0]['lr']:.2e}",
                        f"{rate:.1f} it/s", f"eta {left/60:.1f}m"]
                # The size of the solution the network is currently producing,
                # per parameter setting. A run can drive its loss down while
                # quietly shrinking the field, and nothing else in this line
                # would show it.
                if self.pinn.last_scale is not None:
                    s = self.pinn.last_scale
                    bits.append(f"|u| {s.min():.2e}..{s.max():.2e}")
                bits += [f"{k} {v:.3e}" for k, v in self.last_metrics.items()
                         if k.startswith("l2/") or k == "holdout/total"]
                print("    " + "  ".join(bits), flush=True)

            postfix = {
                "loss": f"{total.item():.3e}",
                "lr":   f"{self.optimiser.param_groups[0]['lr']:.2e}",
            }
            if self.pinn.last_scale is not None:
                postfix["|u|"] = f"{self.pinn.last_scale.max():.2e}"
                self._check_collapse(step)
            for name, val in self.last_metrics.items():
                if name.startswith("l2/"):
                    postfix[name] = f"{val:.3e}"
                elif name == "holdout/total":
                    postfix["holdout"] = f"{val:.3e}"
            pbar.set_postfix(postfix)

            if step % self.checkpoint_every == 0:
                self._save_checkpoint(step)

        if self.writer is not None:
            self.writer.close()
            
        if self.save_final:
            self._save_checkpoint(self.n_iter)
    
    def train_expanding_horizon(self, T_target, n_stages, n_iter_per_stage=None):
        """
        Expanding-window time curriculum: train on [T_start, T_start + h],
        [T_start, T_start + 2h], ..., [T_start, T_target] across n_stages stages.

        The same network and optimiser are reused — each stage warm-starts from
        the previous one. IC sampling stays at t = T_start throughout. The global
        step counter is continuous, so checkpoints are uniquely named
        (ckpt_{absolute_step}.pt) and the LR scheduler keeps its state.

        Args:
            T_target         : final time of the last stage
            n_stages         : number of curriculum stages
            n_iter_per_stage : int (same iters per stage) |
                               callable(k, N) -> int (k is 1-based) |
                               None (split self.n_iter equally across stages)
        """
        if n_iter_per_stage is None:
            per_stage = max(1, self.n_iter // n_stages)
            n_iter_fn = lambda k, N: per_stage
        elif callable(n_iter_per_stage):
            n_iter_fn = n_iter_per_stage
        else:
            v = int(n_iter_per_stage)
            n_iter_fn = lambda k, N: v

        geo           = self.pinn.sampler.geometry
        T_start       = geo.T[0]
        saved_n_iter  = self.n_iter
        absolute_step = self.start_step

        for k in range(1, n_stages + 1):
            T_now    = T_start + (T_target - T_start) * k / n_stages
            geo.T    = [T_start, T_now]
            self.pinn.points = None

            self.n_iter     = n_iter_fn(k, n_stages)
            self.start_step = absolute_step

            print(f"=== Stage {k}/{n_stages}: T in [{T_start:.4g}, {T_now:.4g}], "
                  f"{self.n_iter} iters ===")
            self.train()
            absolute_step += self.n_iter + 1

        self.n_iter = saved_n_iter
        
    def finetune_lbfgs(self, n_outer=100, n_cycles=1, max_iter=20, lr=1.0,
                       history_size=100, tolerance_grad=1e-8,
                       tolerance_change=1e-12, save_checkpoint=True):
        """
        L-BFGS post-training fine-tuning.

        After Adam/NAdam has reached a good basin, L-BFGS typically drops the
        loss another 1-2 orders of magnitude. The collocation pool is frozen
        within each cycle (L-BFGS Hessian approximation requires a stationary
        loss). If n_cycles > 1, the pool is refreshed via resample_adaptive()
        between cycles and a fresh L-BFGS state is initialised each time.

        Args:
            n_outer        : L-BFGS outer calls per cycle
            n_cycles       : number of L-BFGS phases (with resampling between)
            max_iter       : line-search iterations per outer call
            lr             : initial step length (strong_wolfe rescales it)
            history_size   : L-BFGS memory (typical 50-200)
            tolerance_grad : gradient-norm stop criterion
        """
        if self.pinn.points is None:
            self.pinn.resample()

        for cycle in range(n_cycles):
            self.pinn.resample_adaptive()

            lbfgs = torch.optim.LBFGS(
                self.pinn.net.parameters(),
                lr=lr,
                max_iter=max_iter,
                history_size=history_size,
                tolerance_grad=tolerance_grad,
                tolerance_change=tolerance_change,
                line_search_fn="strong_wolfe",
            )

            state = {"loss": None}
            pbar  = tqdm(range(n_outer),
                         desc=f"L-BFGS cycle {cycle + 1}/{n_cycles}")

            for k in pbar:
                def closure():
                    lbfgs.zero_grad()
                    residuals = self.pinn.step()
                    total, _  = self._aggregate_loss(residuals)
                    total.backward()
                    state["loss"] = total.item()
                    return total

                lbfgs.step(closure)

                if self.logger_type == "tensorboard" and self.writer is not None:
                    self.writer.add_scalar(
                        "loss/total", state["loss"],
                        self.start_step + cycle * n_outer + k,
                    )

                pbar.set_postfix({"loss": f"{state['loss']:.3e}"})

            self.start_step += n_outer

            if save_checkpoint:
                self._save_checkpoint(self.n_iter+self.start_step)
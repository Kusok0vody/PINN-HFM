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
        lra_alpha:        EMA rate for the loss-weight update
        param_every:      resample physics parameters every K iterations (0 = never)
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
        lra_alpha:        float = 0.9,
        param_every:      int   = 0,
        validator               = None,
        validate_every:   int   = 0,
        balancing:        str   = "lra",
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

        if balancing not in ("lra", "none"):
            raise ValueError(
                f"unknown balancing scheme '{balancing}'; choose 'lra' or 'none'"
            )
        self.balancing        = balancing
        self.gradnorm_every   = gradnorm_every
        self.lra_alpha        = lra_alpha

        self.adaptive_weights = {}
        self._weights_initialized = False
        self.param_every = param_every

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

    def _update_weights_lra(self, residuals: dict):
        """
        Learning Rate Annealing, Wang et al., as given in Bischof & Kraus Eq. (6):

            lambda_hat_i(t) = max|grad_theta L_Omega(t)| / mean|grad_theta L_i(t)|
            lambda_i(t)     = alpha lambda_i(t-1) + (1 - alpha) lambda_hat_i(t)

        The governing-equation loss L_Omega is the reference and keeps weight 1;
        only the boundary, initial and data terms are rescaled. Note the
        statistics are a max and a mean over gradient *entries*, not L2 norms —
        the two differ substantially once the parameter count is large.

        Where the paper assumes a single L_Omega, a system contributes several
        PDE residuals; their sum is used as the reference, which is the loss the
        governing equations actually pose.

        Despite its name this schedules the loss weights, not the learning rate.
        """
        params = [p for p in self.pinn.net.parameters() if p.requires_grad]
        losses = self._term_losses(residuals)

        pde_terms = [v for k, v in losses.items() if k.startswith("pde/")]
        if not pde_terms:
            return
        ref = self._flat_grad(sum(pde_terms), params)
        num = ref.abs().max().item()

        for key, loss in losses.items():
            if key.startswith("pde/"):
                self.adaptive_weights[key] = 1.0
                continue
            den = self._flat_grad(loss, params).abs().mean().item()
            if den < 1e-12 or num < 1e-12:
                continue
            lam_hat = num / den
            old     = self.adaptive_weights.get(key, 1.0)
            self.adaptive_weights[key] = (
                self.lra_alpha * old + (1.0 - self.lra_alpha) * lam_hat
            )

        self.optimiser.zero_grad()

    def _update_weights(self, residuals: dict):
        if self.balancing == "lra":
            self._update_weights_lra(residuals)

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
                "scheduler":  self.scheduler.state_dict(),
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
        if scheduler is not None:
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
            
            if self.param_every > 0 and step % self.param_every == 0:
                self.pinn.physics._resample_parameters()
                # The objective just changed; an adapted step size carried
                # across that boundary refers to the previous problem.
                if hasattr(self.optimiser, "reset_lr"):
                    self.optimiser.reset_lr()
            
            if step > self.start_step and step % self.resample_every == 0:
                self.pinn.resample_adaptive()

            self.optimiser.zero_grad()

            residuals = self.pinn.step()

            if self.balancing == "lra":
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
                bits += [f"{k} {v:.3e}" for k, v in self.last_metrics.items()
                         if k.startswith("l2/") or k == "holdout/total"]
                print("    " + "  ".join(bits), flush=True)

            postfix = {
                "loss": f"{total.item():.3e}",
                "lr":   f"{self.optimiser.param_groups[0]['lr']:.2e}",
            }
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
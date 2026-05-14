import os
import torch
from tqdm import tqdm
import datetime


class Trainer:
    """
    Training loop for PINN.

    Responsibilities:
        - optimiser and lr-scheduler
        - loss aggregation from residuals with per-residual weights
        - logging via tqdm + tensorboard
        - checkpointing (net state_dict + optimiser + sampled points)
        - calling resample() every resample_every iterations

    Args:
        pinn:             PINN instance
        mu_list:          list of parameter dicts [{"alpha": 1.0, ...}, ...]
        weights:          per-residual weights, e.g. {"convection": 1.0, "pde": 1.0}
                          lookup order: exact name -> group name -> 1.0
        lr:               initial learning rate
        n_iter:           number of training iterations
        resample_every:   resample collocation points every K iterations
        checkpoint_every: save checkpoint every K iterations
        checkpoint_path:  directory for checkpoints
        logger:           "tqdm" or "tensorboard"
        device:           "cpu" or "cuda"
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
        self.start_step       = start_step

        self.gradnorm_every   = gradnorm_every
        self.lra_alpha        = lra_alpha
        self.adaptive_weights = {
            f"{group}/{name}": 1.0
            for group, res_dict in {}.items()
            for name in res_dict
        }
        self._weights_initialized = False
        self.param_every = param_every

        self.optimiser = torch.optim.NAdam(
            pinn.net.parameters(),
            lr=lr,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimiser,
            patience=1000,
            factor=0.5,
            min_lr=1e-6,
        )

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

    def _update_weights_gradnorm(self, residuals: dict):
        grad_norms = {}

        for group, res_dict in residuals.items():
            for name, res in res_dict.items():
                key  = f"{group}/{name}"
                w    = self.adaptive_weights.get(key, 1.0)
                term = w * (res**2).mean()

                self.optimiser.zero_grad()
                term.backward(retain_graph=True)

                grads = [
                    p.grad.flatten()
                    for p in self.pinn.net.parameters()
                    if p.grad is not None
                ]
                if grads:
                    grad_norms[key] = torch.cat(grads).norm().item()

        self.optimiser.zero_grad()

        if not grad_norms:
            return

        mean_norm = sum(grad_norms.values()) / len(grad_norms)

        for key, norm in grad_norms.items():
            if norm < 1e-12:
                continue
            w_new = mean_norm / (norm + 1e-8)
            w_new = max(0.01, min(w_new, 100.0))
            old   = self.adaptive_weights.get(key, 1.0)
            self.adaptive_weights[key] = (
                (1 - self.lra_alpha) * old + self.lra_alpha * w_new
            )

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
        else:
            pinn.net.load_state_dict(ckpt["net"])
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
        self.pinn.resample()

        pbar = tqdm(range(self.start_step, self.start_step+self.n_iter+1), desc="Training")

        for step in pbar:
            
            if self.param_every > 0 and step % self.param_every == 0:
                self.pinn.physics._resample_parameters()
            
            if step > self.start_step and step % self.resample_every == 0:
                self.pinn.resample_adaptive()

            self.optimiser.zero_grad()

            residuals = self.pinn.step()
            if not self._weights_initialized or step % self.gradnorm_every == 0:
                self._update_weights_gradnorm(residuals)
                self._weights_initialized = True
                residuals = self.pinn.step()
                
            total, loss_terms = self._aggregate_loss(residuals)

            total.backward()
            self.optimiser.step()
            self.scheduler.step(total)

            self._log(step, loss_terms, total.item())

            pbar.set_postfix({
                "loss": f"{total.item():.3e}",
                "lr":   f"{self.optimiser.param_groups[0]['lr']:.2e}",
            })

            if step % self.checkpoint_every == 0:
                self._save_checkpoint(step)

        if self.writer is not None:
            self.writer.close()
            
        if self.save_final:
            self._save_checkpoint(self.n_iter)
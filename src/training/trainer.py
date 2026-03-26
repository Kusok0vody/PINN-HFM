import os
import torch
from tqdm import tqdm


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
                          lookup order: exact name → group name → 1.0
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
        weights:          dict  = None,
        lr:               float = 1e-3,
        n_iter:           int   = 10000,
        resample_every:   int   = 1000,
        checkpoint_every: int   = 1000,
        checkpoint_path:  str   = "checkpoints",
        logger:           str   = "tqdm",
        device:           str   = "cpu",
    ):
        self.pinn             = pinn
        self.weights          = weights or {}
        self.n_iter           = n_iter
        self.resample_every   = resample_every
        self.checkpoint_every = checkpoint_every
        self.checkpoint_path  = checkpoint_path
        self.device           = device
        self.logger_type      = logger

        self.mu = pinn.physics.set_par().to(device)

        self.optimiser = torch.optim.NAdam(
            pinn.net.parameters(),
            lr=lr,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimiser,
            patience=500,
            factor=0.5,
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
        else:
            self.writer = None

    def _aggregate_loss(self, residuals: dict) -> tuple[torch.Tensor, dict]:
        """
        Aggregates residuals into scalar loss using weights.

        Lookup order for weight: exact residual name → group name → 1.0

        Returns:
            total loss tensor, dict of individual loss term values
        """
        loss_terms = {}
        total      = torch.tensor(0.0, device=self.device)

        for group, res_dict in residuals.items():
            for name, res in res_dict.items():
                w    = self.weights.get(name, self.weights.get(group, 1.0))
                term = w * (res ** 2).mean()
                loss_terms[f"{group}/{name}"] = term.item()
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
                "step":      step,
                "net":       self.pinn.net.state_dict(),
                "optimiser": self.optimiser.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "mu":        self.mu,
                "points":    self.pinn.points,
            },
            f"{self.checkpoint_path}/ckpt_{step}.pt",
        )

    def load_checkpoint(self, path: str) -> int:
        """
        Loads checkpoint and restores training state.

        Returns:
            step number from checkpoint
        """
        ckpt = torch.load(path, map_location=self.device)
        self.pinn.net.load_state_dict(ckpt["net"])
        self.optimiser.load_state_dict(ckpt["optimiser"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.mu          = ckpt["mu"]
        self.pinn.points = ckpt["points"]
        return ckpt["step"]

    def train(self):
        self.pinn.net.to(self.device)
        self.pinn.resample()

        pbar = tqdm(range(self.n_iter), desc="Training")

        for step in pbar:

            if step > 0 and step % self.resample_every == 0:
                self.pinn.resample()

            self.optimiser.zero_grad()

            residuals         = self.pinn.step(self.mu)
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
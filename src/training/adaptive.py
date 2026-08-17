import torch
from torch.optim import Optimizer


class HyperGradAdam(Optimizer):
    """
    Adam whose learning rate is itself updated by gradient descent.

    Baydin, Cornish, Martinez-Rubio, Schmidt, Wood (2018), "Online Learning
    Rate Adaptation with Hypergradient Descent" — the Adam-HD variant.

    The loss after a step depends on the step size that produced it, and that
    dependence is differentiable:

        d f(theta_t) / d alpha = <grad f(theta_t), d theta_t / d alpha>
                               = -<g_t, u_{t-1}>

    where u_{t-1} is the Adam direction actually applied at the previous step.
    Descending on alpha therefore gives

        alpha_t = alpha_{t-1} + beta <g_t, u_{t-1}>

    Consecutive steps that agree push the rate up; steps that undo each other —
    the signature of overshooting — pull it down.

    Two properties matter here, both in contrast to a line search:

      * no extra loss evaluations. Only the gradient already computed for the
        weight update is used, so a PINN pays nothing for re-differentiating
        residuals.
      * no comparison of loss values across steps. The rate is nudged rather
        than accepted or rejected, so an objective that shifts underneath it —
        a physics parameter resample, a loss-weight update — perturbs the rate
        instead of invalidating a decision.

    Args:
        params:    iterable of parameters
        lr:        initial rate; the method is reported to be insensitive to it
        betas:     Adam moment decay rates
        eps:       Adam denominator guard
        hyper_lr:  beta, the rate at which the learning rate itself moves
        normalize: use the cosine of (g_t, u_{t-1}) instead of their raw inner
                   product. This is NOT in the paper. The additive rule's beta
                   carries the units of the gradient squared, so a value that
                   works on one problem is wrong on the next — exactly the
                   tuning this is meant to remove. The cosine makes beta
                   dimensionless and hyper_lr a relative rate.
        lr_min,
        lr_max:    keep the adapted rate in a sane range
    """

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999),
                 eps: float = 1e-8, hyper_lr: float = 0.3,
                 normalize: bool = True, lr_min: float = 1e-8,
                 lr_max: float = 1.0):
        super().__init__(params, dict(lr=lr, lr0=lr, betas=betas, eps=eps,
                                      hyper_lr=hyper_lr, normalize=normalize,
                                      lr_min=lr_min, lr_max=lr_max))
        self._k = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._k += 1

        for group in self.param_groups:
            b1, b2 = group["betas"]
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue

            # --- hypergradient: how the last step's length affected this loss
            num = dot = gn = un = 0.0
            for p in params:
                st = self.state[p]
                if "u_prev" in st:
                    dot += (p.grad * st["u_prev"]).sum().item()
                    gn  += p.grad.pow(2).sum().item()
                    un  += st["u_prev"].pow(2).sum().item()
            if dot != 0.0 or gn > 0.0:
                if group["normalize"]:
                    denom = (gn ** 0.5) * (un ** 0.5)
                    num = dot / denom if denom > 1e-16 else 0.0
                    group["lr"] *= (1.0 + group["hyper_lr"] * num)
                else:
                    group["lr"] += group["hyper_lr"] * dot
                group["lr"] = min(max(group["lr"], group["lr_min"]), group["lr_max"])

            # --- ordinary Adam step with the freshly adapted rate
            for p in params:
                st = self.state[p]
                if "m" not in st:
                    st["m"] = torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p)
                st["m"].mul_(b1).add_(p.grad, alpha=1 - b1)
                st["v"].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                mhat = st["m"] / (1 - b1 ** self._k)
                vhat = st["v"] / (1 - b2 ** self._k)
                u = mhat / (vhat.sqrt() + group["eps"])
                p.add_(u, alpha=-group["lr"])
                st["u_prev"] = u

        return loss

    def reset_lr(self, lr: float = None):
        """
        Return the rate to its starting value.

        The hypergradient reads the alignment of consecutive gradients, which
        assumes both come from the same objective. When the physics parameters
        are resampled the objective changes outright: the error jumps, the
        alignment becomes noise, and the rate inherited from before the change
        no longer refers to anything. Measured on CDR, leaving it alone let the
        rate drift two orders of magnitude upward, into a range where training
        collapses.
        """
        for group in self.param_groups:
            group["lr"] = group["lr0"] if lr is None else lr

    @property
    def last_eta(self) -> float:
        return self.param_groups[0]["lr"]

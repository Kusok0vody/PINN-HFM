"""
Does the field survive the first thousand steps?

A 30000-step run ended at exactly u = 0: residual 2e-9, boundary error exactly
1, amplitude ratio 0.000 at every k. The checkpoint trace put the collapse
inside the first 1000 steps, which is the whole window that matters — after
that there is nothing left to observe, because u = 0 solves a linear
homogeneous equation exactly and, under a multiplicative output scale, every
gradient is proportional to the output and dies with it.

So this runs only that window, on the three configurations worth telling apart:

    detached   the divisor of the scale-free residual cut from the graph, which
               is what ran overnight. Its gradient is d/dg (r/s) = r/s, a push
               towards a smaller field that does not weaken as the field
               shrinks, while the boundary term that would pull back weakens
               like |u|.
    attached   the same divisor keeping its graph, so the ratio is homogeneous
               of degree zero and the equation term has no opinion on amplitude
               at all. This is what the code now does.
    no gain    scale-free residual without the multiplicative output scale. Here
               u = 0 is not a fixed point: the last layer's gradient there is
               -2*mean(g_data * h), which has no reason to vanish, so the run
               can climb back out of a dip.

What to look for is the |u| column, not the loss. A collapsing run has a
falling amplitude and an improving loss at the same time — that is exactly what
makes it hard to notice.

Reference: an unflagged baseline shrinks by a factor of a few over these steps
and recovers. A factor of a thousand is the failure.

Usage:
    python experiments/helmholtz/collapse_probe.py            # ~10 min on an A100
    python experiments/helmholtz/collapse_probe.py --steps 400
"""
import argparse, math, pathlib, sys

import torch
import torch.nn as nn

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "src"))
sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from geometry.geom import Geometry
from geometry.sampler import Sampler
from network.net import Net
from network.activations import ActivationFactory, Sine
from physics.problems.helmholtz import helmholtz2D_annulus
from pinn import PINN
from training.trainer import Trainer
from eval_helmholtz import build_bounds

ARMS = [
    ("detached  (what collapsed)", True,  True,  True),
    ("attached  (current code)",   True,  True,  False),
    ("no gain,  scale-free only",  False, True,  False),
    ("neither   (baseline)",       False, False, False),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--n-points", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    bounds = build_bounds()
    samp   = Sampler(Geometry(bounds, dim=2, has_time=False),
                     n_interior=args.n_points, n_boundary=64)
    orig   = PINN._field_scale

    for tag, gain, scale_free, detached in ARMS:
        torch.manual_seed(args.seed)
        net = Net(x_dim=2, mu_dim=1, dx=32, dmu=64, d_h=64,
                  encoder_layers=3, trunk_layers=4, head_layers=3, film_layers=1,
                  activation=ActivationFactory(Sine, omega=1.0, trainable=False),
                  encoder_activation=nn.Tanh,
                  trunk_activation=ActivationFactory(Sine, omega=1.0, trainable=False),
                  head_activation=nn.Tanh, film_activation=nn.Tanh,
                  outputs_config={"u": {"activation":
                                        ActivationFactory(Sine, omega=1.0, trainable=True)}},
                  use_film=True, use_fourier=False, output_scaling=gain)
        ph = helmholtz2D_annulus(dim=2, has_time=False, device=device)
        ph.setParameters(params=[{"k": float(k)} for k in range(1, 11)],
                         boundaries=bounds)
        pinn = PINN(net, ph, samp, n_refine=10, adaptive_pde=True,
                    paired_coords=True, scale_free_pde=scale_free, device=device)

        # The overnight run cut the divisor from the graph. Restoring that here
        # is the only way to compare against it rather than against a guess.
        PINN._field_scale = ((lambda self, pred, detach=True: orig(self, pred, True))
                             if detached else orig)
        tr = Trainer(pinn=pinn, n_iter=args.steps, lr=1e-3, resample_every=1000,
                     gradnorm_every=200, lra_alpha=0.01, logger="none",
                     progress=False, log_every=max(1, args.steps // 12),
                     checkpoint_every=10 ** 9, save_final=False, device=device)
        print(f"\n=== {tag}", flush=True)
        tr.train()
        PINN._field_scale = orig
        print(f"    final |u| {pinn.last_scale.min():.3e}..{pinn.last_scale.max():.3e}"
              f"   started at {tr._amp0:.3e}", flush=True)


if __name__ == "__main__":
    main()

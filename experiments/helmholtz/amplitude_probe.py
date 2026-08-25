"""
Is the amplitude ceiling the objective's doing, or the optimiser's?

Training on the annulus saturates at a solution whose rms amplitude is about
1.95, whatever the exact solution asks for: the projection onto the reference
obeys alpha = min(1, 1.93 / A) across thirteen parameter settings spanning both
sides of two Dirichlet eigenvalues, with the shape right everywhere. The ceiling
survived three learning-rate schedules — constant 1e-3 gave 1.24, plateau ending
at 2.5e-4 gave 1.93, cosine ending at 1e-5 gave 1.96 — so below a threshold the
rate does not move it.

That leaves two explanations and no way to tell them apart from a training
curve:

    the objective   the weighted loss is genuinely lowest at this amplitude, and
                    the optimiser found what it was asked for.
    the optimiser   the loss would be lower at a larger amplitude, and training
                    did not get there.

They are separated by one number. Freeze the trained network, multiply its
output by a scalar s, and evaluate the same loss the training minimised as a
function of s. If the minimum sits at s = 1 the objective is content and the
ceiling is the objective's; if it sits well above 1 the objective wants a larger
field and the ceiling is the optimiser's.

No training, no gradients through weights, one scalar swept over a grid.

Usage:
    python experiments/helmholtz/amplitude_probe.py --ckpt checkpoints/helmholtz
"""
import argparse, math, pathlib, sys

import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "src"))
sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from geometry.geom import Geometry
from geometry.sampler import Sampler
from network.net import Net
from physics.problems.helmholtz import helmholtz2D_annulus
from pinn import PINN
from training.trainer import Trainer
from eval_helmholtz import build_bounds, resolve_checkpoint

R, r, N_ARCS = 3.0, 1.0, 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--k-min", type=float, default=1.0)
    ap.add_argument("--k-max", type=float, default=20.0)
    ap.add_argument("--n-k", type=int, default=32)
    ap.add_argument("--n-points", type=int, default=4096)
    ap.add_argument("--n-data", type=int, default=256)
    ap.add_argument("--hard-bc", action="store_true",
                    help="the checkpoint was trained with the hard boundary "
                         "ansatz. Without this the network output is read as "
                         "the solution when it is only the free part of it, "
                         "and every number below is wrong without saying so.")
    ap.add_argument("--scales", type=float, nargs="+",
                    default=[0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0])
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    path   = resolve_checkpoint(args.ckpt)
    net    = Net.from_checkpoint(str(path), device=device)
    net.eval()
    print(f"checkpoint {path}")

    bounds  = build_bounds()
    samp    = Sampler(Geometry(bounds, dim=2, has_time=False),
                      n_interior=args.n_points, n_boundary=64)
    physics = helmholtz2D_annulus(dim=2, has_time=False, device=device,
                              hard_bc=args.hard_bc)
    physics.setParameters(
        params=[{"k": float(v)} for v in
                torch.linspace(args.k_min, args.k_max, args.n_k)],
        boundaries=bounds,
        limits={"k": {"min": args.k_min, "max": args.k_max, "N": args.n_k}},
    )
    pinn = PINN(net, physics, samp, n_refine=1, adaptive_pde=True,
                scale_free_pde=True, autoscale_inputs=False, device=device)
    pinn.set_data(None, n_points=args.n_data, seed=42)

    # The weights are the ones the run ended with, so the balancing weights it
    # ended with are the ones that define "the loss it was minimising". They are
    # in the checkpoint; without them this would be scoring a different
    # objective from the one that produced the network.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    trainer = Trainer(pinn=pinn, n_iter=1, logger="none", progress=False,
                      checkpoint_every=10 ** 9, save_final=False, device=device)
    stored = ckpt.get("weights")
    if stored:
        trainer.adaptive_weights = dict(stored)
        print(f"using the run's own balancing weights ({len(stored)} terms)")
    else:
        print("checkpoint has no stored weights; scoring with all weights at 1")

    # One pool of points for every scale, so the comparison is between scales
    # and not between draws.
    pinn.resample()

    # Per setting, not globally. A single multiplier over the whole batch would
    # be dominated by the settings that are already right: the ceiling only
    # bites where the exact solution is large, and those are a minority of any
    # sweep. Each parameter setting occupies its own column of every residual,
    # and the loss is a sum over columns, so the question "what amplitude would
    # this setting prefer" has a separate answer for each one.
    ks = physics.par.tensor.detach().cpu().reshape(-1).tolist()
    base = dict(physics.transforms)
    per_setting = {}

    for scale in args.scales:
        physics.transforms = {n: (lambda u, f=f, s=scale: s * f(u))
                              for n, f in base.items()}
        res = pinn.step()
        for m in range(len(ks)):
            tot = 0.0
            by = {}
            for group, terms in res.items():
                # The equation term is homogeneous of degree zero in the field
                # by construction, so it contributes the same at every scale and
                # is left out of the comparison rather than added identically to
                # both sides of it.
                if group == "pde":
                    continue
                for key, v in terms.items():
                    w = trainer.adaptive_weights.get(f"{group}/{key}", 1.0)
                    c = w * float(v[:, m].detach().pow(2).mean())
                    tot += c
                    by[group] = by.get(group, 0.0) + c
            per_setting.setdefault(m, []).append((scale, tot, by))
    physics.transforms = base

    print()
    # A minimum one grid step away is not evidence of anything: what matters is
    # how much lower the loss would be there. A tenth of a percent is noise; the
    # near-pole settings need the amplitude to grow four to ten times, and a
    # gain that small at 1.2x says the objective is not asking for it.
    GAIN_MIN = 0.05

    print(f"{'k':>8} {'best s':>7} {'gain':>7} {'bc(1)':>10} {'bc(best)':>10}"
          f" {'data(1)':>10} {'data(best)':>10}   {'verdict':>12}")
    n_up = 0
    for m, rows in sorted(per_setting.items()):
        one = next(r for r in rows if r[0] == 1.0)
        best = min(rows, key=lambda t: t[1])
        gain = (one[1] - best[1]) / max(one[1], 1e-30)
        if best[0] > 1.0 and gain > GAIN_MIN:
            n_up += 1
            verdict = "wants more"
        elif best[0] < 1.0 and gain > GAIN_MIN:
            verdict = "wants less"
        else:
            verdict = "content"
        print(f"{ks[m]:>8.3f} {best[0]:>7.2f} {100*gain:>6.1f}% "
              f"{one[2].get('bc', 0):>10.3e} {best[2].get('bc', 0):>10.3e} "
              f"{one[2].get('data', 0):>10.3e} {best[2].get('data', 0):>10.3e}"
              f"   {verdict:>12}")

    print()
    if n_up == 0:
        print("No setting would lower the loss by growing. The ceiling is the "
              "objective's: it is already at the amplitude it asked for, and "
              "further training cannot move it. What has to change is the loss.")
    else:
        print(f"{n_up} of {len(ks)} settings would lower the loss at a larger "
              f"amplitude than training reached. To that extent the ceiling is "
              f"the optimiser's, and the objective is not what is holding it.")


if __name__ == "__main__":
    main()

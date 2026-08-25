"""
Which of two networks does the objective actually prefer?

A run started from a supervised fit that was right almost everywhere — the
projection onto the exact solution was 0.99 at k = 6.3 and the boundary was out
by an rms of 0.13, four times better than anything the PINN objective reaches on
its own — and left it within a thousand steps, ending at 0.06. That says the
weighted loss is lower somewhere else. It is worth checking, because there is a
second possibility that looks identical from the training curve: the balancing
weights move while training runs, so "the objective" is not one function. A
point can be worse under the weights a run began with and better under the ones
it ended with, in which case the loss did not prefer it — the rule redefining
the loss did.

So the same two networks are scored twice: once with every term weighted
equally, and once with the weights the run finished with. Equal weights are the
only fixed objective available, and if the good network wins under them while
losing under the trained ones, the balancing is what moved.

Everything else is held constant — same collocation pool, same data points, same
parameter settings — so the only difference between the two columns is the
network, and between the two rows the weights.

Usage:
    python experiments/helmholtz/loss_compare.py \\
        --a checkpoints/warm.pt --b checkpoints/helmholtz/ckpt_20000.pt
"""
import argparse, pathlib, sys

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


def score(pinn, trainer, weights):
    """Total loss and its per-group parts, under a given set of weights."""
    trainer.adaptive_weights = dict(weights) if weights else {}
    res = pinn.step()
    total, terms = trainer._aggregate_loss(res)
    by = {}
    for key, v in terms.items():
        by[key.split("/")[0]] = by.get(key.split("/")[0], 0.0) + v
    return float(total), by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="first checkpoint, e.g. the warm start")
    ap.add_argument("--b", required=True, help="second, e.g. where training ended")
    ap.add_argument("--k-min", type=float, default=1.0)
    ap.add_argument("--k-max", type=float, default=20.0)
    ap.add_argument("--n-k", type=int, default=32)
    ap.add_argument("--n-points", type=int, default=4096)
    ap.add_argument("--n-data", type=int, default=1024)
    ap.add_argument("--hard-bc", action="store_true",
                    help="the checkpoint was trained with the hard boundary "
                         "ansatz. Without this the network output is read as "
                         "the solution when it is only the free part of it, "
                         "and every number below is wrong without saying so.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    bounds = build_bounds()
    samp   = Sampler(Geometry(bounds, dim=2, has_time=False),
                     n_interior=args.n_points, n_boundary=64)

    physics = helmholtz2D_annulus(dim=2, has_time=False, device=device,
                              hard_bc=args.hard_bc)
    physics.setParameters(
        params=[{"k": float(v)} for v in
                torch.linspace(args.k_min, args.k_max, args.n_k)],
        boundaries=bounds,
        limits={"k": {"min": args.k_min, "max": args.k_max, "N": args.n_k}},
    )
    physics._resample_parameters()

    # One pool, one data set, drawn once and reused for both networks, so the
    # comparison is between networks and not between draws.
    torch.manual_seed(args.seed)
    shared_points = samp.sample().to(device)

    trained_weights = None
    results = {}
    for tag, spec in (("A", args.a), ("B", args.b)):
        path = resolve_checkpoint(spec)
        net  = Net.from_checkpoint(str(path), device=device)
        pinn = PINN(net, physics, samp, n_refine=1, adaptive_pde=True,
                    scale_free_pde=True, device=device)
        pinn.points = shared_points
        pinn.set_data(None, n_points=args.n_data, seed=42)
        trainer = Trainer(pinn=pinn, n_iter=1, logger="none", progress=False,
                          checkpoint_every=10 ** 9, save_final=False, device=device)

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if ckpt.get("weights"):
            trained_weights = ckpt["weights"]
        results[tag] = (pinn, trainer, path)

    if trained_weights is None:
        print("neither checkpoint stores balancing weights; only equal weights "
              "can be compared")

    print()
    print(f"A = {results['A'][2]}")
    print(f"B = {results['B'][2]}")
    print()
    print(f"{'weights':>16} {'A total':>12} {'B total':>12} {'prefers':>9}")
    for label, w in (("equal", None),
                     ("run's final", trained_weights)):
        if w is None and label != "equal":
            continue
        a_tot, a_by = score(*results["A"][:2], w)
        b_tot, b_by = score(*results["B"][:2], w)
        print(f"{label:>16} {a_tot:>12.4e} {b_tot:>12.4e} "
              f"{('A' if a_tot < b_tot else 'B'):>9}")
        for g in sorted(set(a_by) | set(b_by)):
            print(f"{'  ' + g:>16} {a_by.get(g, 0):>12.4e} {b_by.get(g, 0):>12.4e}")

    print()
    print("A winning under equal weights and losing under the run's own weights "
          "means the balancing moved the target. A losing under both means the "
          "objective genuinely prefers where training went, and the loss is what "
          "has to change.")


if __name__ == "__main__":
    main()

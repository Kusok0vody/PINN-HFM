"""
Can this network represent the solution at all, physics aside?

Training saturates at a field whose interior is about twice its boundary values,
whatever the exact solution asks for. Near a Dirichlet eigenvalue the exact
solution is amplified far past that — at k = 6.3 its rms is 7.6 against a
boundary datum of 1 — and the projection onto it falls to 0.25 while the shape
stays right and the equation's residual grows.

Three explanations remain and the training curve cannot separate them:

    the objective    the loss is lowest where training stopped. Ruled out: a
                     scalar sweep on the frozen network finds its minimum at
                     exactly the amplitude reached, at 28 of 32 settings.
    the optimiser    the loss would be lower elsewhere and training did not get
                     there. Weakened: three learning-rate schedules gave the
                     same ceiling once the rate was small enough.
    the architecture the field simply cannot be represented by this network.

This tests the third directly, by deleting the first two. No PDE, no boundary
conditions, no balancing — plain regression of the network onto the exact
solution, at the same points, with the same optimiser. If it reaches the
solution, the architecture is capable and the fault is in the objective or the
optimisation. If it saturates at the same place, the fault is the architecture,
and width and depth are what to change.

The comparison to keep in mind: in training, alpha reached 0.98 at k = 1 and
0.25 at k = 6.3.

Usage:
    python experiments/helmholtz/capacity_probe.py
    python experiments/helmholtz/capacity_probe.py --d-h 128 --steps 8000
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
from validation.references import helmholtz_annulus
from eval_helmholtz import build_bounds, decompose

R, r, N_ARCS = 3.0, 1.0, 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=float, nargs="+", default=[1.0, 6.3, 10.0, 14.5])
    ap.add_argument("--n-points", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dx", type=int, default=32)
    ap.add_argument("--dmu", type=int, default=64)
    ap.add_argument("--d-h", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", default=None, metavar="PATH",
                    help="write the fitted network as a checkpoint, so a PINN "
                         "run can start from it. That turns this probe into the "
                         "one experiment that separates a bad landscape from a "
                         "bad objective: put the network where the answer is "
                         "and see whether the full loss keeps it there.")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    # The same architecture the runs use, so that a limit found here is the
    # limit they were up against.
    net = Net(
        x_dim=2, mu_dim=1, dx=args.dx, dmu=args.dmu, d_h=args.d_h,
        encoder_layers=3, trunk_layers=4, head_layers=3, film_layers=1,
        activation=ActivationFactory(Sine, omega=1.0, trainable=False),
        encoder_activation=nn.Tanh,
        trunk_activation=ActivationFactory(Sine, omega=1.0, trainable=False),
        head_activation=nn.Tanh, film_activation=nn.Tanh,
        outputs_config={"u": {"activation": ActivationFactory(Sine, omega=1.0,
                                                             trainable=True)}},
        use_film=True, use_fourier=False,
    ).to(device)
    n_par = sum(p.numel() for p in net.parameters())
    print(f"net dx={args.dx} dmu={args.dmu} d_h={args.d_h}, {n_par} weights")

    samp = Sampler(Geometry(build_bounds(), dim=2, has_time=False),
                   n_interior=args.n_points, n_boundary=64)
    coords = samp.sample().interior.coords
    y, x = coords[:, 0].numpy(), coords[:, 1].numpy()

    mu = torch.tensor([[k] for k in args.k], dtype=torch.float32)
    target = torch.tensor(
        [[helmholtz_annulus(x, y, float(k), r, R, N_ARCS)] for k in args.k],
        dtype=torch.float32,
    ).squeeze(1).T.contiguous()
    print(f"{coords.shape[0]} points, {len(args.k)} settings, "
          f"reference rms {[round(float(target[:, j].pow(2).mean().sqrt()), 2) for j in range(len(args.k))]}")

    # The same input rescaling the runs use, so the network sees what it is
    # used to. Without it the coordinates arrive at their raw scale and the
    # comparison would be against a differently conditioned problem.
    lows, highs = samp._bounding_box()
    net.set_input_bounds(x_lo=lows, x_hi=highs,
                         mu_lo=torch.tensor([min(args.k)]),
                         mu_hi=torch.tensor([max(args.k)]),
                         mu_log=torch.zeros(1, dtype=torch.bool))

    coords, target, mu = coords.to(device), target.to(device), mu.to(device)
    opt = torch.optim.NAdam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps,
                                                       eta_min=1e-5)

    # Relative per setting, exactly as the data term in training is, so a
    # setting whose solution is twenty times larger does not simply take the
    # whole loss.
    scale = target.pow(2).mean(dim=0, keepdim=True).sqrt().clamp_min(1e-6)

    print()
    print(f"{'step':>7} {'loss':>11}   " +
          "  ".join(f"a(k={k:g})" for k in args.k))
    for step in range(args.steps + 1):
        opt.zero_grad()
        pred = net(coords, mu)["u"]
        loss = ((pred - target) / scale).pow(2).mean()
        loss.backward()
        opt.step()
        sched.step()

        if step % max(1, args.steps // 10) == 0:
            with torch.no_grad():
                a = [decompose(pred[:, j].cpu(), target[:, j].cpu())[0]
                     for j in range(len(args.k))]
            print(f"{step:>7} {loss.item():>11.4e}   " +
                  "  ".join(f"{v:8.3f}" for v in a), flush=True)

    with torch.no_grad():
        pred = net(coords, mu)["u"]
    print()
    print(f"{'k':>8} {'alpha':>8} {'shape':>8}   verdict")
    for j, k in enumerate(args.k):
        al, sh = decompose(pred[:, j].cpu(), target[:, j].cpu())
        v = "represented" if abs(al - 1) < 0.15 and sh < 0.25 else "NOT reached"
        print(f"{k:>8.2f} {al:>8.3f} {sh:>8.3f}   {v}")
    if args.save:
        out = pathlib.Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"step": 0, "run_name": "capacity_probe",
                    "net": net.state_dict(),
                    "net_config": net.serialize_config(),
                    "optimiser": opt.state_dict(),
                    "scheduler": None, "points": None, "weights": None,
                    "balancing": "none"}, out)
        print("")
        print(f"saved to {out}")

    print()
    print("alpha near 1 everywhere means the architecture can hold these fields "
          "and the ceiling seen in training belongs to the objective or the "
          "optimisation. alpha falling off at the amplified settings means the "
          "architecture is the limit.")


if __name__ == "__main__":
    main()

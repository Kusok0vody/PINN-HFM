"""
How well does a trained Lotka-Volterra run solve the family it was swept over?

Eleven parameters means the settings cannot be listed by hand the way a single
wavenumber could, so this draws them: a fresh sample from the same limits the
run was trained on, under its own seed. Those settings were never trained on —
the sweep redraws every hundred steps and never returns to a point — so every
number below is generalisation within the family rather than recall of it.

Reported per variable, because G, H and P are not equally hard: the predator
population is the smallest and the most damped, and an error that is invisible
in the grass is the whole of P.

The error split is the same one used for Helmholtz. Writing u = alpha*ref + r
with r orthogonal to ref,

    L2^2 = (alpha - 1)^2 + shape^2

exactly, so a run that has the shape of the solution but the wrong size is
distinguishable from one that has neither, which a single L2 cannot do.

Usage:
    python experiments/lotka_volterra/eval_lotka_volterra.py \\
        --ckpt checkpoints/lotka_volterra
    ... --plot figures --n-settings 64
"""
import argparse
import pathlib
import sys

import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "src"))
sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from geometry.geom                   import Geometry
from geometry.sampler                import Sampler
from network.net                     import Net
from physics.problems.lotka_volterra import LotkaVolterra
from pinn                            import PINN

import sweep

VARS = ("G", "H", "P")


def decompose(u, ref):
    """
    Split the error into a scale part and a shape part.

    alpha is the least-squares multiple of the reference present in the
    prediction and carries its sign; shape is what is left once that best
    scalar multiple has been removed. The two satisfy
    L2^2 = (alpha - 1)^2 + shape^2 exactly.

    Keeping them apart matters here for the same reason it did on the annulus:
    a population that decays to near zero has a reference of small norm, and a
    prediction that is right in shape but off by a constant factor reads as a
    large relative error with no indication of which of the two went wrong.
    """
    denom = ref.pow(2).sum()
    if denom <= 0:
        return float("nan"), float("nan"), float("nan")
    alpha = float((u * ref).sum() / denom)
    shape = float((u - alpha * ref).norm() / ref.norm())
    l2    = float((u - ref).norm() / ref.norm())
    return l2, alpha, shape


def resolve_checkpoint(spec):
    """
    Accept either a file or a directory of ckpt_<step>.pt.

    Sorting by name would put ckpt_9900 after ckpt_20000 and sorting by mtime
    would pick whichever file the filesystem touched last, so the step number
    is parsed and compared as a number.
    """
    path = pathlib.Path(spec)
    if path.is_file():
        return path
    if not path.is_dir():
        raise SystemExit(f"no such checkpoint or directory: {spec}")

    found = []
    for p in path.glob("ckpt_*.pt"):
        try:
            found.append((int(p.stem.split("_")[-1]), p))
        except ValueError:
            continue
    if not found:
        raise SystemExit(f"{path} holds no ckpt_<step>.pt files")

    step, p = max(found)
    steps = sorted(s for s, _ in found)
    print(f"{len(found)} checkpoints in {path}, steps {steps[0]}..{steps[-1]}; "
          f"using step {step}")
    return p


def build(args, device):
    """Network, physics and PINN, with the evaluation settings installed."""
    path = resolve_checkpoint(args.ckpt)
    net  = Net.from_checkpoint(str(path), device=device)
    net.eval()
    step = torch.load(path, map_location="cpu", weights_only=False)["step"]
    print(f"checkpoint {path}, step {step}")

    bounds  = sweep.bounds()
    limits  = sweep.limits(args.n_settings)
    geo     = Geometry(bounds, dim=1, has_time=False)
    samp    = Sampler(geo, n_interior=args.n_t, n_boundary=0)
    physics = LotkaVolterra(dim=1, has_time=False,
                            hard_ic=not args.soft_ic, device=device)
    physics.setParameters(params=[sweep.midpoint()], boundaries=bounds,
                          limits=limits, initial=None)

    if net.config["mu_dim"] != len(limits):
        raise SystemExit(
            f"the checkpoint was trained on {net.config['mu_dim']} parameters "
            f"and this sweep declares {len(limits)}. One of them has been "
            f"edited since; nothing below would mean anything."
        )

    # Drawn, not pinned. The pinned ends are the settings every training batch
    # contains by construction, so scoring on them would measure the one part
    # of the range the run has seen most, and the interior is what a parametric
    # solution is for.
    torch.manual_seed(args.seed)
    physics.par = physics.make_param_batch(
        physics.draw_parameters(args.n_settings, anchor_ends=False))

    # The checkpoint carries the input rescaling it was trained under, and
    # recomputing it here would feed the weights coordinates they never saw.
    pinn = PINN(net, physics, samp, scale_free_pde=True,
                autoscale_inputs=False, device=device)
    return pinn, physics, net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="a ckpt_<step>.pt, or the directory holding them")
    ap.add_argument("--n-settings", type=int, default=32,
                    help="how many parameter settings to draw and score")
    ap.add_argument("--n-t", type=int, default=1001,
                    help="points on the time axis")
    ap.add_argument("--seed", type=int, default=1234,
                    help="which settings get drawn; change it to check that a "
                         "result is not one lucky sample")
    ap.add_argument("--worst", type=int, default=5,
                    help="how many of the worst settings to print in full")
    ap.add_argument("--time-bands", type=int, default=5,
                    help="split the horizon into this many bands and report "
                         "the error in each; an initial value problem earns "
                         "its error over time and a single number hides that")
    ap.add_argument("--soft-ic", action="store_true",
                    help="the checkpoint was trained without the hard initial "
                         "condition. Without this the ansatz is applied to a "
                         "network that was never trained under it, and every "
                         "number below is wrong without saying so.")
    ap.add_argument("--plot", nargs="?", const="figures", default=None,
                    metavar="DIR", help="write trajectory figures")
    ap.add_argument("--plot-n", type=int, default=4,
                    help="how many settings to draw figures for")
    ap.add_argument("--device",
                    default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    pinn, physics, net = build(args, device)
    mu = physics.par.tensor

    t = torch.linspace(0.0, sweep.T_END, args.n_t,
                       device=device).reshape(-1, 1)
    with torch.no_grad():
        pred = pinn.predict(t, mu)
    ref = {k: v.to(device) for k, v in physics.reference(t.cpu(), mu.cpu()).items()}

    # --- the initial condition, which the ansatz is supposed to make exact ---
    print()
    ic = physics.initial_values()
    print("initial condition (hard ansatz makes this identical, ~1e-7 in float32)")
    for v in VARS:
        d = (pred[v][0] - ic[v]).abs().max()
        print(f"  {v}: max |u(0) - {physics.IC_PARAMS[v]}| = {float(d):.3e}")

    # --- per variable, over all settings ---
    rows = {v: [] for v in VARS}
    for v in VARS:
        for m in range(mu.shape[0]):
            rows[v].append(decompose(pred[v][:, m], ref[v][:, m]))

    def q(xs, p):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(p * len(xs)))]

    print()
    print(f"{args.n_settings} settings drawn from the sweep, seed {args.seed}")
    print(f"{'var':>4} {'median L2':>10} {'p90 L2':>9} {'worst L2':>9}"
          f" {'median a':>9} {'median shape':>13}")
    for v in VARS:
        l2 = [r[0] for r in rows[v]]
        al = [r[1] for r in rows[v]]
        sh = [r[2] for r in rows[v]]
        print(f"{v:>4} {q(l2, .5):>10.4f} {q(l2, .9):>9.4f} {max(l2):>9.4f}"
              f" {q(al, .5):>9.4f} {q(sh, .5):>13.4f}")

    # --- where in time the error lives ---
    print()
    print("relative L2 by time band, pooled over settings")
    edges = torch.linspace(0.0, sweep.T_END, args.time_bands + 1)
    header = "".join(f"{f'{float(edges[i]):.0f}-{float(edges[i+1]):.0f}':>10}"
                     for i in range(args.time_bands))
    print(f"{'var':>4}{header}")
    for v in VARS:
        cells = []
        for i in range(args.time_bands):
            sel = (t.reshape(-1) >= edges[i]) & (t.reshape(-1) <= edges[i + 1])
            u, r = pred[v][sel], ref[v][sel]
            cells.append(float((u - r).norm() / r.norm().clamp_min(1e-30)))
        print(f"{v:>4}" + "".join(f"{c:>10.4f}" for c in cells))

    # --- the settings that went worst, with the parameters that produced them ---
    total = [max(rows[v][m][0] for v in VARS) for m in range(mu.shape[0])]
    order = sorted(range(len(total)), key=lambda m: -total[m])
    print()
    print(f"worst {min(args.worst, len(order))} settings by the worst of their "
          f"three variables")
    for m in order[:args.worst]:
        pars = ", ".join(f"{k}={float(mu[m, i]):.3f}"
                         for i, k in enumerate(physics.param_order))
        per = "  ".join(f"{v} {rows[v][m][0]:.3f} (a {rows[v][m][1]:+.3f})"
                        for v in VARS)
        print(f"  {per}")
        print(f"      {pars}")

    # --- the equation itself, on the same points ---
    unpacked, pr = pinn._evaluate(t, mu)
    res = physics.residualPDE(pr, unpacked)
    print()
    print("PDE residual rms, and the size of the field it is measured against")
    for key, r in res.items():
        v = key.split("_")[0]
        print(f"  {key:>6}: {float(r.detach().pow(2).mean().sqrt()):.4e}"
              f"   |{v}| = {float(ref[v].pow(2).mean().sqrt()):.4e}")

    if args.plot:
        plot(args, physics, pred, ref, t, mu, order)


def plot(args, physics, pred, ref, t, mu, order):
    """Trajectories for a few settings: prediction against reference."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = pathlib.Path(args.plot)
    outdir.mkdir(parents=True, exist_ok=True)

    # The worst settings first: a figure of the cases that already work says
    # less than a figure of the ones that do not.
    picks = order[:args.plot_n]
    tt = t.reshape(-1).cpu()

    fig, axes = plt.subplots(len(picks), 1, figsize=(9, 3 * len(picks)),
                             squeeze=False)
    for ax, m in zip(axes[:, 0], picks):
        for v, c in zip(VARS, ("tab:green", "tab:blue", "tab:red")):
            ax.plot(tt, ref[v][:, m].cpu(), color=c, lw=2, alpha=.35,
                    label=f"{v} reference")
            ax.plot(tt, pred[v][:, m].detach().cpu(), color=c, lw=1.2,
                    ls="--", label=f"{v} network")
        pars = ", ".join(f"{k}={float(mu[m, i]):.2f}"
                         for i, k in enumerate(physics.param_order))
        ax.set_title(pars, fontsize=7)
        ax.set_xlabel("t")
        ax.legend(fontsize=6, ncol=3)
    fig.tight_layout()
    path = outdir / "lotka_volterra_worst.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

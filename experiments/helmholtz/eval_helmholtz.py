"""
What did the finished Helmholtz run actually learn?

The training log reports a weighted sum of seventeen residual terms. That number
is not an error: the weights move during training, so it is not even comparable
with itself across steps, let alone across runs. This script answers the
question the log cannot.

It reports three things, deliberately separated:

  reference error   relative L2 against the analytic annulus solution. The only
                    number that says whether the answer is right.

  PDE residual      RMS of laplace(u) + k u on the interior. Reference-free, and
                    the same quantity the training minimises — evaluated on
                    fresh points, so it is out of sample.

  boundary error    max and RMS of u - g on each ring, separately for the inner
                    ring (u = 0) and the outer one (u = +/-1). A run that fits
                    the PDE and misses the boundary looks exactly like a run
                    that does the opposite in the total loss, and the two call
                    for opposite fixes.

Error is also broken down by radius. Near the outer ring the boundary datum is
discontinuous, so both the network and the truncated Bessel series oscillate
there; a large error confined to that band means something different from a
large error spread over the interior.

Usage:
    python experiments/helmholtz/eval_helmholtz.py --ckpt checkpoints/ckpt_final.pt
    python experiments/helmholtz/eval_helmholtz.py --ckpt ... --k 1 5 10
"""
import argparse, math, pathlib, sys

import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from geometry.geom import Geometry
from geometry.sampler import Sampler
from network.net import Net
from physics.problems.helmholtz import helmholtz2D_annulus
from pinn import PINN
from validation.metrics import relative_l2, max_abs_error
from validation.references import helmholtz_annulus, helmholtz_annulus_resonances

R, r, N_ARCS = 3.0, 1.0, 8


def build_bounds():
    """The same geometry the training script builds — kept in step by construction."""
    bounds = {}
    for i in range(N_ARCS):
        val = 1.0 if i % 2 == 0 else -1.0
        for tag, rad, fn in (
            ("rq", r, lambda x, y: torch.zeros_like(x)),
            ("Rq", R, (lambda x, y, v=val: v * torch.ones_like(x))),
        ):
            bounds[f"{tag}{i+1}"] = {
                "p": [i / N_ARCS, (i + 1) / N_ARCS],
                "x": (lambda p, a=rad: a * torch.cos(2 * math.pi * p)),
                "y": (lambda p, a=rad: a * torch.sin(2 * math.pi * p)),
                "N": 64,
                "bc": {"u": {"type": "dirichlet", "value": fn}},
            }
    return bounds


def pde_residual(pinn, coords, mu, chunk):
    """
    Residual of the equation on a grid, in chunks.

    Two things make this the memory hot spot of the whole script. The residual
    is second order, so its graph is retained through create_graph, and it is
    evaluated on a plotting grid — tens of thousands of points, an order more
    than a training batch. At ten parameter settings that combination reached
    69 GB on an 80 GB card in one go.

    Chunking bounds it at whatever a chunk costs, and going through
    pinn._evaluate rather than unpack_coords means the residual is computed by
    the same path the training loop uses, paired coordinates included.
    """
    out = []
    for i in range(0, coords.shape[0], chunk):
        unpacked, pred = pinn._evaluate(coords[i:i + chunk], mu)
        res = pinn.physics.residualPDE(pred, unpacked)["helmholtz"]
        out.append(res.detach().cpu())
        del unpacked, pred, res
    return torch.cat(out, dim=0)


def checkpoint_series(spec, every):
    """All ckpt_<step>.pt under a directory, in step order, every Nth."""
    path = pathlib.Path(spec)
    if not path.is_dir():
        raise SystemExit(f"--history needs the checkpoint directory, got {spec}")
    found = []
    for p in path.glob("ckpt_*.pt"):
        try:
            found.append((int(p.stem.split("_")[-1]), p))
        except ValueError:
            continue
    return sorted(found)[::every]


def resolve_checkpoint(spec):
    """
    Accept either a file or a directory of ckpt_<step>.pt.

    Sorting by name would put ckpt_9900 after ckpt_20000, and sorting by mtime
    would pick whichever file the filesystem touched last, so the step number is
    parsed and compared as a number.
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
    others  = sorted(s for s, _ in found)
    print(f"{len(found)} checkpoints in {path}, steps {others[0]}..{others[-1]}; "
          f"using step {step}")
    return p


def polar_grid(n_rho, n_theta, device):
    rho = torch.linspace(r, R, n_rho)
    th  = torch.linspace(0, 2 * math.pi, n_theta + 1)[:-1]
    RHO, TH = torch.meshgrid(rho, th, indexing="ij")
    x, y = RHO * torch.cos(TH), RHO * torch.sin(TH)
    # Geometry order for (has_time=False, dim=2) is (y, x).
    coords = torch.stack([y.reshape(-1), x.reshape(-1)], 1).to(device)
    return coords, RHO.reshape(-1), x.reshape(-1), y.reshape(-1)


def ring(rad, n, device):
    th = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
    x, y = rad * torch.cos(th), rad * torch.sin(th)
    return torch.stack([y, x], 1).to(device), th


def plot_comparison(pinn, k, n, outdir, device):
    """
    Network, exact solution and their difference, side by side.

    Everything outside the annulus is masked. The network does produce values
    in the hole, but there is no equation there, no collocation points and no
    boundary condition, so those values mean nothing and drawing them invites
    reading a pattern into pure extrapolation.

    The first two panels share one colour scale, so their colours are directly
    comparable; the difference gets its own symmetric scale, since it is
    typically an order of magnitude smaller and would otherwise be a uniform
    green square.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    g = np.linspace(-R, R, n)
    X, Y = np.meshgrid(g, g, indexing="xy")
    rho = np.hypot(X, Y)
    inside = (rho >= r) & (rho <= R)

    exact = helmholtz_annulus(X, Y, k, r, R, N_ARCS)          # NaN outside already

    coords = torch.stack(
        [torch.as_tensor(Y.ravel(), dtype=torch.float32),
         torch.as_tensor(X.ravel(), dtype=torch.float32)], 1
    ).to(device)
    mu = torch.tensor([[k]], dtype=torch.float32, device=device)
    with torch.no_grad():
        u = pinn.predict(coords, mu)["u"][:, 0].cpu().numpy().reshape(X.shape)
    u = np.where(inside, u, np.nan)

    diff = u - exact
    lim  = np.nanmax(np.abs(exact))
    dlim = np.nanmax(np.abs(diff))
    th   = np.linspace(0, 2 * math.pi, 400)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    panels = [
        (u,     "PINN",                       "jet",    -lim,  lim),
        (exact, "exact",                       "jet",    -lim,  lim),
        (diff,  "PINN - exact",                "RdBu_r", -dlim, dlim),
    ]
    for ax, (field, title, cmap, lo, hi) in zip(axes, panels):
        m = ax.pcolormesh(X, Y, field, cmap=cmap, vmin=lo, vmax=hi, shading="auto")
        for rad in (r, R):
            ax.plot(rad * np.cos(th), rad * np.sin(th), "k", lw=0.8)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("x")
        fig.colorbar(m, ax=ax, fraction=0.046)
    axes[0].set_ylabel("y")
    fig.suptitle(f"k = {k}   max|exact| = {lim:.3f}   max|error| = {dlim:.3f}")
    fig.tight_layout()

    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"helmholtz_k{k:g}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def history(args, device, bounds, samp, physics):
    """
    Amplitude ratio and relative L2 through training, from saved checkpoints.

    The final checkpoint cannot say whether the field was pushed down early and
    climbed back, or simply never got there — and with a scale-free residual
    those are different diagnoses with different fixes. Only the predictions are
    needed here, so no residual and no figures: one forward pass per checkpoint.
    """
    series = checkpoint_series(args.ckpt, args.history)
    if not series:
        raise SystemExit(f"no checkpoints under {args.ckpt}")

    coords, rho, gx, gy = polar_grid(args.n_rho, args.n_theta, device)
    mu   = torch.tensor([[k] for k in args.k], dtype=torch.float32, device=device)
    refs = [torch.as_tensor(helmholtz_annulus(gx.numpy(), gy.numpy(), k, r, R, N_ARCS),
                            dtype=torch.float32) for k in args.k]
    ref_rms = [q.pow(2).mean().sqrt() for q in refs]

    print(f"{len(series)} checkpoints, steps {series[0][0]}..{series[-1][0]}")
    print("amplitude ratio (rms PINN / rms exact), then relative L2\n")
    head = "".join(f"{'k=' + f'{k:g}':>10}" for k in args.k)
    print(f"{'step':>7}{head}   |{head}")

    for step, path in series:
        net  = Net.from_checkpoint(str(path), device=device)
        net.eval()
        pinn = PINN(net, physics, samp, autoscale_inputs=False,
                    paired_coords=True, device=device)
        with torch.no_grad():
            pred = pinn.predict(coords, mu)["u"].cpu()
        amp = [f"{pred[:, j].pow(2).mean().sqrt() / ref_rms[j]:10.3f}"
               for j in range(len(args.k))]
        l2  = [f"{relative_l2(pred[:, j], refs[j]):10.3f}"
               for j in range(len(args.k))]
        print(f"{step:>7}{''.join(amp)}   |{''.join(l2)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="a ckpt_<step>.pt file, or the directory holding them, "
                         "in which case the largest step is used")
    ap.add_argument("--k", type=float, nargs="+", default=[10.0],
                    help="parameter settings to evaluate; must match what was trained")
    ap.add_argument("--n-rho", type=int, default=80)
    ap.add_argument("--n-theta", type=int, default=256)
    ap.add_argument("--plot", nargs="?", const="figures", default=None,
                    help="also write a PINN / exact / difference figure per k, "
                         "into this directory (default 'figures')")
    ap.add_argument("--plot-n", type=int, default=401,
                    help="cartesian resolution of the figure")
    ap.add_argument("--jump-bins", type=int, default=6,
                    help="bins of the outer-ring error profile, from a jump in "
                         "the boundary datum to the middle of an arc")
    ap.add_argument("--res-chunk", type=int, default=2048,
                    help="points per chunk when computing the PDE residual; the "
                         "second-order graph is what fills the card, so lower "
                         "this before lowering the grid")
    ap.add_argument("--history", type=int, default=0, metavar="EVERY",
                    help="instead of one report, trace amplitude ratio and "
                         "relative L2 across every EVERY-th checkpoint in the "
                         "directory. Answers whether the field recovers after "
                         "an early dip, which a final checkpoint cannot.")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    # A single k gives a degenerate interval, so scan a window around the
    # request and report the distance instead: what matters is not whether a
    # resonance is bracketed but how close the nearest one is, since the
    # solution amplitude grows without bound as it is approached.
    near = helmholtz_annulus_resonances(max(1e-6, min(args.k) - 2.0),
                                        max(args.k) + 2.0, r, R, N_ARCS)
    for k in args.k:
        close = [v for v in near if abs(v - k) < 1.0]
        if close:
            print(f"warning: k = {k} is {min(abs(v - k) for v in close):.3f} from a "
                  f"Dirichlet eigenvalue {[round(v, 3) for v in close]} — the exact "
                  f"solution is amplified there and the errors below will be large "
                  f"for reasons that have nothing to do with training")

    bounds = build_bounds()
    geo    = Geometry(bounds, dim=2, has_time=False)
    samp   = Sampler(geo, n_interior=1024, n_boundary=64)

    physics = helmholtz2D_annulus(dim=2, has_time=False, device=device)
    physics.setParameters(params=[{"k": k} for k in args.k], boundaries=bounds)

    if args.history > 0:
        return history(args, device, bounds, samp, physics)

    path = resolve_checkpoint(args.ckpt)
    net  = Net.from_checkpoint(str(path), device=device)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    print(f"checkpoint {path}, step {ckpt['step']}, run '{ckpt.get('run_name')}'")

    # The checkpoint carries its own input rescaling; recomputing it here would
    # overwrite the map the weights were trained under whenever this script's
    # sampler or limits differ from the training script's by anything at all.
    # Paired coordinates for the residual: one backward instead of one per
    # setting, which is the difference between fitting on the card and not.
    pinn = PINN(net, physics, samp, autoscale_inputs=False,
                paired_coords=True, device=device)

    coords, rho, gx, gy = polar_grid(args.n_rho, args.n_theta, device)
    mu = torch.tensor([[k] for k in args.k], dtype=torch.float32, device=device)

    # The residual needs the graph on the coordinates, so it cannot sit inside
    # no_grad; the reference error can and does.
    res = pde_residual(pinn, coords, mu, args.res_chunk)

    with torch.no_grad():
        pred = pinn.predict(coords, mu)["u"].cpu()

    inner, th_i = ring(r, 512, device)
    outer, th_o = ring(R, 512, device)
    with torch.no_grad():
        u_in  = pinn.predict(inner, mu)["u"].cpu()
        u_out = pinn.predict(outer, mu)["u"].cpu()
    g_out = torch.where(torch.sin((N_ARCS // 2) * th_o) >= 0, 1.0, -1.0).unsqueeze(1)

    arc       = 2 * math.pi / N_ARCS
    to_jump   = (th_o + arc / 2) % arc - arc / 2      # signed distance to nearest jump
    # Equal-width bins from a jump to the middle of an arc. Equal width rather
    # than equal count so that the horizontal axis is angle and the profile can
    # be read as a decay rate.
    bin_edges = torch.linspace(0.0, arc / 2, args.jump_bins + 1)

    bands = [(r, 1.5), (1.5, 2.5), (2.5, 2.9), (2.9, R)]
    print()
    for j, k in enumerate(args.k):
        ref = torch.as_tensor(
            helmholtz_annulus(gx.numpy(), gy.numpy(), k, r, R, N_ARCS),
            dtype=torch.float32,
        )
        u = pred[:, j]
        print(f"k = {k}")
        print(f"  relative L2          {relative_l2(u, ref):.4f}")
        # How much of the solution's size the network actually produced. A PINN
        # that damps the field keeps a small residual — the equation is linear,
        # so any multiple of the solution satisfies it — and pays only at the
        # boundary. Relative L2 mixes that failure together with getting the
        # shape wrong; this separates it out.
        print(f"  amplitude ratio      "
              f"{u.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt():.4f}"
              f"   (1.0 = right size, < 1 = damped)")
        print(f"  max abs error        {max_abs_error(u, ref):.4f}"
              f"   (reference amplitude {ref.abs().max():.3f})")
        print(f"  PDE residual RMS     {res[:, j].pow(2).mean().sqrt():.4e}")
        print(f"  inner ring |u|       max {u_in[:, j].abs().max():.4f}"
              f"  rms {u_in[:, j].pow(2).mean().sqrt():.4f}      (target 0)")
        e_out = u_out[:, j] - g_out[:, 0]
        print(f"  outer ring |u - g|   max {e_out.abs().max():.4f}"
              f"  rms {e_out.pow(2).mean().sqrt():.4f}      (target +/-1)")
        # Where on the ring that error sits. The outer datum steps between +1
        # and -1 at n_arcs points, and no smooth function follows a step, so a
        # band around each jump is wrong by O(1) whatever the training did.
        # Spread over the whole ring that alone yields an rms of a few tenths,
        # which the line above cannot tell apart from a boundary the network
        # genuinely failed to fit.
        #
        # Reported as a profile rather than as a masked number on purpose: a
        # cutoff chosen after seeing the result decides the answer by itself.
        # Here nothing is excluded and no threshold is picked — the shape
        # answers the question. Concentrated at 0 and decaying: the jumps, and
        # nothing to fix. Flat across the arc: a real underfit of the boundary,
        # present at every k including those where the solution is of size one
        # and nothing is being damped.
        print("  outer ring |u - g| by angular distance to the nearest jump:")
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            m = (to_jump.abs() >= lo) & (to_jump.abs() < hi)
            if m.sum() == 0:
                continue
            e = e_out[m]
            print(f"    {math.degrees(lo):5.1f}-{math.degrees(hi):4.1f} deg  "
                  f"n {int(m.sum()):4d}   rms {e.pow(2).mean().sqrt():.4f}"
                  f"   max {e.abs().max():.4f}")
        print("  by radius:")
        for lo, hi in bands:
            m = (rho >= lo) & (rho < hi + (1e-6 if hi == R else 0.0))
            if m.sum() == 0:
                continue
            print(f"    {lo:.1f} <= rho < {hi:.1f}   rel L2 {relative_l2(u[m], ref[m]):.4f}"
                  f"   max err {max_abs_error(u[m], ref[m]):.4f}")
        if args.plot is not None:
            print(f"  figure               "
                  f"{plot_comparison(pinn, k, args.plot_n, args.plot, device)}")
        print()


if __name__ == "__main__":
    main()

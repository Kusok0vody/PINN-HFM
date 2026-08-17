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
from utils import unpack_coords
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="a ckpt_<step>.pt file, or the directory holding them, "
                         "in which case the largest step is used")
    ap.add_argument("--k", type=float, nargs="+", default=[10.0],
                    help="parameter settings to evaluate; must match what was trained")
    ap.add_argument("--n-rho", type=int, default=80)
    ap.add_argument("--n-theta", type=int, default=256)
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

    path = resolve_checkpoint(args.ckpt)
    net  = Net.from_checkpoint(str(path), device=device)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    print(f"checkpoint {path}, step {ckpt['step']}, run '{ckpt.get('run_name')}'")

    bounds = build_bounds()
    geo    = Geometry(bounds, dim=2, has_time=False)
    samp   = Sampler(geo, n_interior=1024, n_boundary=64)

    physics = helmholtz2D_annulus(dim=2, has_time=False, device=device)
    physics.setParameters(params=[{"k": k} for k in args.k], boundaries=bounds)

    # The checkpoint carries its own input rescaling; recomputing it here would
    # overwrite the map the weights were trained under whenever this script's
    # sampler or limits differ from the training script's by anything at all.
    pinn = PINN(net, physics, samp, autoscale_inputs=False, device=device)

    coords, rho, gx, gy = polar_grid(args.n_rho, args.n_theta, device)
    mu = torch.tensor([[k] for k in args.k], dtype=torch.float32, device=device)

    # The residual needs the graph on the coordinates, so it cannot sit inside
    # no_grad; the reference error can and does.
    unpacked, coords_g = unpack_coords(coords, False, 2, requires_grad=True)
    pred_g = physics.apply_output_ansatz(
        physics.apply_transforms(net(coords_g, mu)), unpacked
    )
    res = physics.residualPDE(pred_g, unpacked)["helmholtz"].detach().cpu()

    with torch.no_grad():
        pred = pinn.predict(coords, mu)["u"].cpu()

    inner, th_i = ring(r, 512, device)
    outer, th_o = ring(R, 512, device)
    with torch.no_grad():
        u_in  = pinn.predict(inner, mu)["u"].cpu()
        u_out = pinn.predict(outer, mu)["u"].cpu()
    g_out = torch.where(torch.sin((N_ARCS // 2) * th_o) >= 0, 1.0, -1.0).unsqueeze(1)

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
        print(f"  max abs error        {max_abs_error(u, ref):.4f}"
              f"   (reference amplitude {ref.abs().max():.3f})")
        print(f"  PDE residual RMS     {res[:, j].pow(2).mean().sqrt():.4e}")
        print(f"  inner ring |u|       max {u_in[:, j].abs().max():.4f}"
              f"  rms {u_in[:, j].pow(2).mean().sqrt():.4f}      (target 0)")
        e_out = u_out[:, j] - g_out[:, 0]
        print(f"  outer ring |u - g|   max {e_out.abs().max():.4f}"
              f"  rms {e_out.pow(2).mean().sqrt():.4f}      (target +/-1)")
        print("  by radius:")
        for lo, hi in bands:
            m = (rho >= lo) & (rho < hi + (1e-6 if hi == R else 0.0))
            if m.sum() == 0:
                continue
            print(f"    {lo:.1f} <= rho < {hi:.1f}   rel L2 {relative_l2(u[m], ref[m]):.4f}"
                  f"   max err {max_abs_error(u[m], ref[m]):.4f}")
        print()


if __name__ == "__main__":
    main()

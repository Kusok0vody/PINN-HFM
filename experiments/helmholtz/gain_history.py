"""
What did the amplitude head learn, and was it still learning when training
stopped?

The evaluation says the field ends five to seven per cent too large almost
everywhere, and the amplitude probe says the objective would be lower at 0.95 —
both terms of it, not one fighting the other. That is a gap between where the
loss has its minimum and where training stopped, and it is one scalar per
setting wide, which Adam should close in a few dozen steps at any learning rate
the schedule ever reaches.

So either the gain never moved, or it moved and was still moving. The two have
opposite consequences: a gain that plateaued early is being held by something,
and a gain still drifting at the last checkpoint means the run was simply cut
short and the fix is more steps, not more machinery.

Prints the learned gain per parameter setting across the checkpoint series,
with its per-checkpoint drift, so the shape of the curve is visible rather than
inferred.

Usage:
    python experiments/helmholtz/gain_history.py --ckpt checkpoints/helmholtz
"""
import argparse, pathlib, sys

import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "src"))
sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from network.net import Net
from eval_helmholtz import checkpoint_series


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--every", type=int, default=2000,
                    help="stride through the series, in training steps")
    ap.add_argument("--k", type=float, nargs="+",
                    default=[1.0, 5.9, 10.0, 13.87, 14.5, 15.1, 20.0],
                    help="settings to report, one column each")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    # Stride by training step rather than by position in the series, so the
    # table reads the same whatever checkpoint_every the run used, and always
    # ends on the last checkpoint — the drift on the final row is the number
    # this whole script exists to report.
    paths  = checkpoint_series(args.ckpt, 1)
    if not paths:
        raise SystemExit(f"no checkpoint series under {args.ckpt}")

    mu = torch.tensor([[k] for k in args.k], dtype=torch.float32, device=device)

    rows = []
    for step, path in paths:
        if step % args.every and step != paths[-1][0]:
            continue
        net = Net.from_checkpoint(str(path), device=device)
        if net.magnitude is None:
            raise SystemExit(
                f"{path} was trained without the amplitude head — there is no "
                f"gain to report. Nothing is wrong with the run; this script "
                f"simply does not apply to it."
            )
        net.eval()
        with torch.no_grad():
            z = net.encoder_mu(net._rescale_mu(mu))
            rows.append((step, torch.exp(net.magnitude(z))[:, 0].tolist()))

    head = "".join(f"{k:>9.2f}" for k in args.k)
    print(f"{'step':>7}{head}{'  max drift':>12}")
    prev = None
    for step, g in rows:
        # Drift since the previous reported checkpoint, as a fraction. This is
        # the number the question turns on: a run that has converged in this
        # coordinate reports drift falling toward zero, and one that was cut
        # short reports drift that is still substantial at the last row.
        drift = ("" if prev is None else
                 f"{max(abs(a - b) / max(b, 1e-12) for a, b in zip(g, prev)):>11.1%}")
        print(f"{step:>7}" + "".join(f"{v:>9.4f}" for v in g) + f"{drift:>12}")
        prev = g

    if len(rows) > 1:
        first, last = rows[0][1], rows[-1][1]
        print()
        print("total change over the run: " +
              "  ".join(f"k={k:g} {l/f:.2f}x" for k, f, l in zip(args.k, first, last)))


if __name__ == "__main__":
    main()

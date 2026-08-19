"""
Balancing weights through training, read back from checkpoints.

A run that goes wrong under an automatic weighting scheme leaves no trace of why
in the loss curve: the loss is the weighted sum, so a weight running away and a
residual running away look identical there. This prints the weights themselves.

Nothing here is specific to one problem — it reads whatever terms a run had.
The question it exists for on this project is which terms outvote which: a
boundary described as sixteen arcs enters the objective as sixteen terms, eight
of them a homogeneous condition that is satisfied early and keeps its weight,
and the balancing rules in the literature are formulated for two to four.

Reads whatever the checkpoints carry. Checkpoints written before the weights
were saved say so rather than reporting zeros.

Usage:
    python experiments/weights_history.py checkpoints/helmholtz
    python experiments/weights_history.py checkpoints/helmholtz --every 20
"""
import argparse, pathlib, sys

import torch

# The checkpoints pickle the sampled points, whose classes live under src,
# so the package has to be importable even though nothing here uses it.
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir")
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--top", type=int, default=4,
                    help="how many of the largest and smallest terms to name")
    args = ap.parse_args()

    found = []
    for p in pathlib.Path(args.ckpt_dir).glob("ckpt_*.pt"):
        try:
            found.append((int(p.stem.split("_")[-1]), p))
        except ValueError:
            continue
    if not found:
        raise SystemExit(f"no ckpt_<step>.pt under {args.ckpt_dir}")
    series = sorted(found)[::args.every]

    print(f"{len(series)} checkpoints, steps {series[0][0]}..{series[-1][0]}")
    header = f"{'step':>7} {'n':>3} {'min':>9} {'median':>9} {'max':>9} {'spread':>9}"
    print(header + "   largest terms")
    for step, path in series:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        w  = ck.get("weights")
        if not w:
            print(f"{step:>7}   (no weights stored in this checkpoint)")
            continue
        vals = sorted(w.values())
        med  = vals[len(vals) // 2]
        top  = sorted(w.items(), key=lambda kv: -kv[1])[:args.top]
        names = ", ".join(f"{k}={v:.3g}" for k, v in top)
        print(f"{step:>7} {len(vals):>3} {vals[0]:9.3g} {med:9.3g} {vals[-1]:9.3g} "
              f"{vals[-1] / max(vals[0], 1e-30):9.3g}   {names}")


if __name__ == "__main__":
    main()

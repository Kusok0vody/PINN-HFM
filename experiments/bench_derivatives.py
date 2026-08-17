"""
Time and memory of the two derivative paths, measured properly on a GPU.

Why this exists. The choice between them was settled on a laptop CPU, where the
memory figure came from the process working set and the largest cases were
swapping — the timings there overstate the gap. On a GPU there is no swap, the
allocator reports exactly how much was used, and exceeding the card is a hard
failure rather than a slowdown. That makes the interesting question not "how
much faster" but "at what M does each path stop running at all".

The two paths:

  loop    coordinates shared across parameter settings, so f[n, m] depends on
          the single leaf x[n]. One backward per setting, and at second order
          each of those differentiates through all the first-order graphs, so
          the retained graphs nest rather than add.

  paired  coordinates replicated per setting, so every (point, setting) pair
          owns its leaf and one backward collects the whole diagonal Jacobian.
          One graph per derivative order, whatever M is.

Both compute the same numbers — verified elsewhere to exact agreement on values
and to 3e-7 on the gradients that reach the weights.

Usage:
    python experiments/bench_derivatives.py
    python experiments/bench_derivatives.py --n-points 4096 --m-max 64
    python experiments/bench_derivatives.py --device cpu --reps 1

Every measurement is logged as it happens and appended to a JSONL file, so a
run killed by the scheduler still leaves everything it managed to measure.
"""
import argparse, json, math, pathlib, sys, time

import torch
import torch.nn as nn

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import utils
from network.net import Net
from network.activations import ActivationFactory, Sine
from utils import unpack_coords, unpack_coords_paired


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_net(args, device):
    torch.manual_seed(0)
    net = Net(
        x_dim=2, mu_dim=3,
        dx=args.dx, dmu=args.dx, d_h=args.d_h,
        encoder_layers=2, trunk_layers=args.trunk_layers,
        head_layers=2, film_layers=1,
        activation=ActivationFactory(Sine, omega=1.0, trainable=True),
        encoder_activation=nn.Tanh, film_activation=nn.Tanh,
        outputs_config={"u": {"activation": nn.Tanh}},
        use_film=True, use_fourier=False,
    ).to(device)
    return net


def measure(net, path, n_points, m, order, device, reps):
    """
    One configuration. Returns time per repetition and peak memory, or an OOM
    marker. The peak is read from the CUDA allocator where available, which
    counts tensors rather than process pages and therefore does not include the
    interpreter, the driver context or anything else sharing the card.
    """
    cuda = device.type == "cuda"
    coords = torch.rand(n_points, 2, device=device)
    mu = torch.rand(m, 3, device=device)

    def once():
        if path == "paired":
            un, X = unpack_coords_paired(coords, True, 1, m)
            f = net.forward_paired(X, mu)["u"]
            d = utils.derivative_paired(f, un["x"], order)
        else:
            un, c = unpack_coords(coords, True, 1, requires_grad=True)
            f = net(c, mu)["u"]
            d = utils._derivative_loop(f, un["x"], order)
        (d ** 2).mean().backward()
        net.zero_grad(set_to_none=True)

    try:
        once()                                   # warm-up: allocator and kernels
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)

        t0 = time.time()
        for _ in range(reps):
            once()
        if cuda:
            torch.cuda.synchronize()
        dt = (time.time() - t0) / reps

        # Only CUDA reports a per-measurement peak. The process working set on
        # CPU is a high-water mark that never falls, so a second path measured
        # in the same process would simply inherit the first one's peak and the
        # ratio would read 1.0 no matter what. Better to report nothing than a
        # number that looks like a result.
        peak = torch.cuda.max_memory_allocated(device) / 2 ** 20 if cuda else None
        return {"ok": True, "ms": dt * 1000, "peak_mb": peak}

    except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
        kind = "OOM" if "out of memory" in str(exc).lower() else type(exc).__name__
        return {"ok": False, "err": kind, "detail": str(exc)[:160]}
    finally:
        if cuda:
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-points", type=int, default=2048,
                    help="collocation points, the other scaling axis")
    ap.add_argument("--m-max", type=int, default=64,
                    help="largest number of parameter settings to try")
    ap.add_argument("--orders", type=int, nargs="+", default=[1, 2],
                    help="derivative orders; the gap between paths grows with this")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--dx", type=int, default=32)
    ap.add_argument("--d-h", type=int, default=64)
    ap.add_argument("--trunk-layers", type=int, default=3)
    ap.add_argument("--out", default="results_derivatives.jsonl")
    args = ap.parse_args()

    device = torch.device(args.device)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    log(f"device {device}")
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        log(f"gpu    {p.name}, {p.total_memory / 2**30:.1f} GiB, "
            f"capability {p.major}.{p.minor}")
    log(f"torch  {torch.__version__}")
    if device.type != "cuda":
        log("note   memory is only measured on CUDA; on CPU the process peak "
            "never falls and would read the same for both paths")

    net = build_net(args, device)
    n_par = sum(q.numel() for q in net.parameters())
    log(f"net    dx={args.dx} d_h={args.d_h} trunk={args.trunk_layers}, {n_par} weights")

    ms = [m for m in (1, 2, 4, 8, 16, 32, 64, 128, 256) if m <= args.m_max]
    log(f"sweep  N={args.n_points} points, M in {ms}, orders {args.orders}, "
        f"{args.reps} reps each")
    log(f"out    {out.resolve()}")
    log("")

    # A path that has run out of memory once will not survive a larger M, so it
    # is dropped for the rest of that order rather than retried.
    results = {}
    for order in args.orders:
        dead = set()
        log(f"--- derivative order {order} ---")
        for m in ms:
            for path in ("loop", "paired"):
                if path in dead:
                    log(f"  M={m:4d}  {path:7s}  skipped, already out of memory")
                    continue
                r = measure(net, path, args.n_points, m, order, device, args.reps)
                rec = {"path": path, "order": order, "M": m,
                       "n_points": args.n_points, "device": str(device), **r}
                results[(order, m, path)] = r
                with out.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                if r["ok"]:
                    mem = f"{r['peak_mb']:8.0f} MB" if r["peak_mb"] is not None else "  mem n/a"
                    log(f"  M={m:4d}  {path:7s}  {r['ms']:9.1f} ms   {mem}")
                else:
                    dead.add(path)
                    log(f"  M={m:4d}  {path:7s}  {r['err']}  -> dropping this path")
            a, b = results.get((order, m, "loop")), results.get((order, m, "paired"))
            if a and b and a["ok"] and b["ok"]:
                mem = (f"   x{a['peak_mb']/b['peak_mb']:.1f} memory"
                       if a["peak_mb"] and b["peak_mb"] else "")
                log(f"  M={m:4d}  ratio    x{a['ms']/b['ms']:.1f} time{mem}"
                    f" in favour of paired")
        log("")

    log("=== summary ===")
    for order in args.orders:
        log(f"order {order}:")
        log(f"  {'M':>6s} | {'loop ms':>10s} {'loop MB':>10s} | "
            f"{'paired ms':>10s} {'paired MB':>10s} | {'time':>7s} {'memory':>7s}")
        for m in ms:
            a, b = results.get((order, m, "loop")), results.get((order, m, "paired"))
            if a is None and b is None:
                continue
            fmt = lambda r: (f"{r['ms']:10.1f} " +
                             (f"{r['peak_mb']:10.0f}" if r["peak_mb"] is not None else f"{'n/a':>10s}")
                             if r and r["ok"] else f"{'OOM' if r else 'skipped':>21s}")
            fa, fb = fmt(a), fmt(b)
            rat = ""
            if a and b and a["ok"] and b["ok"]:
                rat = f"x{a['ms']/b['ms']:6.1f} "
                rat += (f"x{a['peak_mb']/b['peak_mb']:6.1f}"
                        if a["peak_mb"] and b["peak_mb"] else f"{'':>7s}")
            log(f"  {m:6d} | {fa} | {fb} | {rat}")
        last_loop = max((m for m in ms if results.get((order, m, "loop"), {}).get("ok")),
                        default=None)
        last_pair = max((m for m in ms if results.get((order, m, "paired"), {}).get("ok")),
                        default=None)
        log(f"  largest M that fits: loop {last_loop}, paired {last_pair}")
        log("")
    log(f"done, {len(results)} measurements written to {out.resolve()}")


if __name__ == "__main__":
    main()

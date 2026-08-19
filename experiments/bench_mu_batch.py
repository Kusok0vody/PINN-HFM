"""
Does a larger parameter batch cost anything in a real training loop?

The isolated derivative benchmark said no: on an A100, second-order derivatives
with the paired path took 8.1 ms at M=4 and 7.6 ms at M=16 — the card is idle at
those sizes, so four times the parameter coverage came free. That was one
kernel, though, with no boundary conditions, no optimiser and no loss balancing.
This runs the actual loop and checks whether the claim survives.

It matters because of a structural argument: coverage of the parameter space has
to come from somewhere, and taking it from a short param_every makes the
objective non-stationary, which breaks every mechanism that estimates anything
from the trajectory — the loss weights, the adaptive point pool, any learning
rate that adapts. Taking it from a larger M leaves the objective stationary. The
only reason not to was cost.

Arms, per problem:

    M=4      what the experiments do today
    M=16     four times the coverage
    M=32     eight times

The loop-versus-paired arms this script started with are gone: the two paths
agreed bitwise on the residual and to 1e-6 on the weight gradients, the paired
one was faster and fitted where the other did not, and the choice was removed
from the code. What is left is the question that choice was made to answer —
whether covering more of the parameter space per step costs anything.

Quality is reported two ways. Helmholtz has an analytic solution at any k, so it
gets a relative L2 averaged over a grid of k. Proppant has no reference, so it
gets the holdout residual: the same residual the training minimises, evaluated
on a frozen pool of points the optimiser never sees. Neither is comparable
across problems; both are comparable across arms.

Usage:
    python experiments/bench_mu_batch.py --problem helmholtz --steps 2000
    python experiments/bench_mu_batch.py --problem proppant  --steps 2000
"""
import argparse, json, math, pathlib, sys, time

import torch
import torch.nn as nn

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from geometry.geom import Geometry
from geometry.sampler import Sampler
from network.net import Net
from network.activations import ActivationFactory, Sine, Morlet
from physics.problems.helmholtz import helmholtz2D_annulus
from physics.problems.proppant import proppantDynamics_dless
from pinn import PINN
from training.trainer import Trainer
from validation.metrics import relative_l2, residual_norms
from validation.references import helmholtz_annulus, helmholtz_annulus_resonances


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- helmholtz

def setup_helmholtz(args, device):
    R, r, n_arcs = 3.0, 1.0, 8
    # The default range stops below the first Dirichlet eigenvalue of this
    # annulus, at k = 6.513. Straddling it looks harmless at small M and is not:
    # the sweep points crowd towards the resonance as M grows, and a solution
    # sitting 0.17 away from it has amplitude 20 against a median of 1.9, so its
    # residual swamps the loss and the comparison stops being about M at all.
    k_min, k_max = args.k_min, args.k_max
    bounds = {}
    for i in range(n_arcs):
        for tag, rad, val in (("rq", r, None), ("Rq", R, 1.0 if i % 2 == 0 else -1.0)):
            bounds[f"{tag}{i+1}"] = {
                "p": [i / n_arcs, (i + 1) / n_arcs],
                "x": (lambda p, a=rad: a * torch.cos(2 * math.pi * p)),
                "y": (lambda p, a=rad: a * torch.sin(2 * math.pi * p)),
                "N": 64,
                "bc": {"u": {"type": "dirichlet",
                             "value": (lambda x, y: torch.zeros_like(x)) if val is None
                                      else (lambda x, y, v=val: v * torch.ones_like(x))}},
            }
    geo  = Geometry(bounds, dim=2, has_time=False)
    samp = Sampler(geo, n_interior=args.n_points, n_boundary=64)

    rho = torch.linspace(r, R, 40)
    th  = torch.linspace(0, 2 * math.pi, 81)[:-1]
    RHO, TH = torch.meshgrid(rho, th, indexing="ij")
    rx, ry = (RHO * torch.cos(TH)).reshape(-1), (RHO * torch.sin(TH)).reshape(-1)
    bad = helmholtz_annulus_resonances(k_min, k_max, r, R, n_arcs)
    if bad:
        log(f"WARNING resonances inside [{k_min}, {k_max}]: "
            f"{[round(v, 3) for v in bad]} — solutions there are unbounded and "
            f"will dominate whichever arm samples closest to them")
    else:
        log(f"sweep k in [{k_min}, {k_max}], no resonances inside")

    grid = torch.linspace(k_min, k_max, 9)
    ref  = [torch.as_tensor(helmholtz_annulus(rx.numpy(), ry.numpy(), k, r, R, n_arcs),
                            dtype=torch.float32) for k in grid.tolist()]
    amps = [float(v.abs().max()) for v in ref]
    log(f"reference amplitude over the evaluation grid: "
        f"median {sorted(amps)[len(amps)//2]:.1f}, max {max(amps):.1f}")

    def make_net(seed):
        torch.manual_seed(seed)
        return Net(x_dim=2, mu_dim=1, dx=32, dmu=32, d_h=64,
                   encoder_layers=2, trunk_layers=3, head_layers=2, film_layers=1,
                   activation=ActivationFactory(Sine, omega=1.0, trainable=False),
                   encoder_activation=nn.Tanh, film_activation=nn.Tanh,
                   outputs_config={"u": {"activation": ActivationFactory(Sine, omega=1.0,
                                                                        trainable=True)}},
                   use_film=True, use_fourier=False)

    def make_physics(m):
        ph = helmholtz2D_annulus(dim=2, has_time=False, device=device)
        ph.setParameters(
            params=[{"k": k.item()} for k in torch.linspace(k_min, k_max, m)],
            boundaries=bounds,
            limits={"k": {"min": k_min, "max": k_max, "N": m, "scale": "linear"}},
        )
        return ph

    def quality(pinn):
        mu = grid.unsqueeze(1).to(device)
        coords = torch.stack([ry, rx], 1).to(device)          # Geometry order is (y, x)
        with torch.no_grad():
            pred = pinn.predict(coords, mu)["u"].cpu()
        prof = [relative_l2(pred[:, j], ref[j]) for j in range(len(grid))]
        return {"metric": "mean relative L2 over k", "value": sum(prof) / len(prof),
                "profile": prof}

    return samp, make_net, make_physics, quality


# ----------------------------------------------------------------- proppant

def setup_proppant(args, device):
    chi = 0.5
    cu, cl = (1 + chi) / 2, (1 - chi) / 2
    nb, nw = 1024, 256
    neumann_p = {"p": {"type": "neumann"}}
    bounds = {
        "lu_wall": {"p": [cu, 1.0], "x": lambda p: 0*p, "y": lambda p: p, "N": nw, "bc": neumann_p},
        "ll_wall": {"p": [0.0, cl], "x": lambda p: 0*p, "y": lambda p: p, "N": nw, "bc": neumann_p},
        "inlet":   {"p": [cl, cu],  "x": lambda p: 0*p, "y": lambda p: p, "N": nb, "bc": {
            "c": {"type": "dirichlet",
                  "value": lambda t, x, y: torch.ones_like(x) * 0.25/0.65 * (t <= 0.5).float()},
            "p": {"type": "neumann", "value": lambda t, x, y: torch.ones_like(x)}}},
        "outlet":  {"p": [0.0, 1.0], "x": lambda p: 0*p + 1, "y": lambda p: p, "N": nb, "bc": {}},
        "bottom":  {"p": [0.0, 1.0], "x": lambda p: p, "y": lambda p: 0*p, "N": nb, "bc": neumann_p},
        "top":     {"p": [0.0, 1.0], "x": lambda p: p, "y": lambda p: 0*p + 1, "N": nb, "bc": neumann_p},
    }
    geo  = Geometry(bounds, dim=2, has_time=True, T=[0.0, 1.0])
    samp = Sampler(geo, n_interior=args.n_points, n_boundary=nb, n_initial=args.n_points)

    rho_f, rho_p, g, H, L = 1.0, 1.2, 0.0, 1.0, 1.0
    p0    = 1 / (12 * 0.01 * L * 1)
    G_par = p0 * H * rho_f * g
    r_par = 0.65 * (rho_p - rho_f) / rho_f
    B_MIN, B_MAX = -2.5, 0.0

    def make_net(seed):
        torch.manual_seed(seed)
        return Net(x_dim=3, mu_dim=4, dx=64, dmu=32, d_h=64,
                   encoder_layers=2, trunk_layers=3, head_layers=3, film_layers=1,
                   activation=nn.Tanh,
                   encoder_activation=ActivationFactory(Sine, omega=0.5, trainable=True),
                   trunk_activation=ActivationFactory(Sine, omega=0.5, trainable=True),
                   head_activation=nn.Tanh, film_activation=nn.Tanh,
                   outputs_config={"c": {"activation": ActivationFactory(Morlet, omega=3.0,
                                                                        trainable=True)},
                                   "px": {}, "py": {}},
                   use_film=True, use_fourier=False)

    def make_physics(m):
        ph = proppantDynamics_dless(dim=2, has_time=True, device=device)
        # beta is the viscosity exponent, the parameter the existing script walks
        # through with a schedule; sweeping it is the natural family here.
        ph.setParameters(
            params=[{"alpha": L / H, "beta": b.item(), "r": r_par, "G": G_par}
                    for b in torch.linspace(B_MIN, B_MAX, m)],
            funcPar={"w": lambda x, y: torch.ones_like(x)},
            boundaries=bounds,
            initial={"c": lambda x, y: torch.zeros_like(x)},
        )
        ph.limits = {"beta": {"min": B_MIN, "max": B_MAX, "N": m, "scale": "linear"}}
        return ph

    def quality(pinn):
        # No reference solution exists, so quality is the residual on a frozen
        # pool the optimiser never trained on.
        from validation.validator import Validator
        val = Validator(pinn, holdout=True, seed=12345)
        m = val.holdout_residual()
        return {"metric": "holdout residual (total)", "value": m.get("holdout/total", float("nan")),
                "profile": {k: v for k, v in m.items() if k != "holdout/total"}}

    return samp, make_net, make_physics, quality


SETUPS = {"helmholtz": setup_helmholtz, "proppant": setup_proppant}


def run_arm(args, device, samp, make_net, make_physics, quality, path, m):
    torch.manual_seed(args.seed)
    net  = make_net(args.seed).to(device)
    ph   = make_physics(m)
    torch.manual_seed(args.seed + 1000)
    pinn = PINN(net, ph, samp, n_refine=1, adaptive_pde=True,
                device=device)
    tr = Trainer(pinn=pinn, lr=1e-3, n_iter=args.steps,
                 resample_every=args.resample_every, checkpoint_every=10**9,
                 gradnorm_every=200, lra_alpha=0.01,
                 param_every=args.param_every, checkpoint_path="/tmp/bench_mu",
                 run_name="mu", save_final=False, logger="none", device=device,
                 progress=False, log_every=max(1, args.steps // 4))
    if device.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    tr.train()
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.time() - t0
    peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    q = quality(pinn)
    return {"wall_s": wall, "ms_per_step": wall / max(args.steps, 1) * 1000,
            "peak_mb": peak, **q}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True, choices=sorted(SETUPS))
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--n-points", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--param-every", type=int, default=0,
                    help="0 keeps the objective stationary, which is the point")
    ap.add_argument("--resample-every", type=int, default=1000)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--k-min", type=float, default=1.0, help="helmholtz sweep, lower end")
    ap.add_argument("--k-max", type=float, default=6.0,
                    help="helmholtz sweep, upper end; the default stops below the "
                         "first resonance of the 1:3 annulus at k = 6.513")
    ap.add_argument("--arms", default="M:4,M:16,M:32")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    out = pathlib.Path(args.out or f"results_mu_{args.problem}.jsonl")

    log(f"problem {args.problem}   device {device}")
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        log(f"gpu     {p.name}, {p.total_memory/2**30:.1f} GiB")
    log(f"steps   {args.steps}, {args.n_points} collocation points, seed {args.seed}")
    log(f"periods param_every={args.param_every}, resample_every={args.resample_every}")

    log("building problem (sampling and any reference solution)...")
    t0 = time.time()
    samp, make_net, make_physics, quality = SETUPS[args.problem](args, device)
    log(f"ready in {time.time()-t0:.1f}s")

    arms = [(a.split(":")[0], int(a.split(":")[1])) for a in args.arms.split(",")]
    log(f"arms    {arms}")
    log(f"out     {out.resolve()}")
    log("")

    results = {}
    for path, m in arms:
        log(f"--- {path}, M={m} ---")
        try:
            r = run_arm(args, device, samp, make_net, make_physics, quality, path, m)
            results[(path, m)] = r
            mem = f"{r['peak_mb']:.0f} MB" if r["peak_mb"] is not None else "n/a"
            log(f"  {r['wall_s']:.1f}s total, {r['ms_per_step']:.1f} ms/step, peak {mem}")
            log(f"  {r['metric']} = {r['value']:.6f}")
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            kind = "OOM" if "out of memory" in str(exc).lower() else type(exc).__name__
            results[(path, m)] = {"error": kind}
            log(f"  FAILED: {kind}  {str(exc)[:120]}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
        with out.open("a") as fh:
            rec = {"problem": args.problem, "path": path, "M": m, "steps": args.steps,
                   "seed": args.seed, **results[(path, m)]}
            fh.write(json.dumps(rec, default=str) + "\n")
        log("")

    log("=== summary ===")
    log(f"  {'arm':>14s} | {'ms/step':>9s} {'peak MB':>9s} | quality")
    ref = results.get(("M", 4))
    for (path, m), r in results.items():
        if "error" in r:
            log(f"  {path+' M='+str(m):>14s} | {r['error']:>19s} |")
            continue
        mem = f"{r['peak_mb']:9.0f}" if r["peak_mb"] is not None else f"{'n/a':>9s}"
        rel = ""
        if ref and "error" not in ref:
            rel = f"   ({r['ms_per_step']/ref['ms_per_step']:.2f}x time vs paired M=4)"
        log(f"  {path+' M='+str(m):>14s} | {r['ms_per_step']:9.1f} {mem} | "
            f"{r['value']:.6f}{rel}")
    log("")
    log("the claim under test: paired M=16 costs about the same per step as M=4")


if __name__ == "__main__":
    main()

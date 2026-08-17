"""
Sweep one setting across seeds and problems, and write the results as JSON lines.

Built for a cluster, where the two limits of the local measurements can finally
be lifted:

  * seeds. Run-to-run spread on these problems is 15-20 percent, so telling a
    20 percent effect from noise needs on the order of 16 seeds per arm — eight
    times what fits on a laptop. Anything measured with two seeds only resolved
    effects of roughly x1.5 and above.
  * problems. Conclusions here do not transfer between problems: the Fourier
    embedding helped by x3.5 on the elliptic Helmholtz annulus and destroyed
    training on transport-dominated convection. A sweep run on one PDE says
    nothing about the others.

Every arm is scored the same way: relative L2 against an analytic reference,
averaged over a grid of the swept physical parameter, so a network that fits
one setting well and interpolates badly cannot look good.

Usage:
    python experiments/benchmark.py --problem cdr --sweep arch --seeds 16
    python experiments/benchmark.py --problem helmholtz --sweep optimiser --seeds 16

Results append to --out as one JSON object per (arm, seed), so array jobs can
write to separate files and be concatenated afterwards.
"""
import argparse, json, math, pathlib, sys, time

import torch
import torch.nn as nn

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from geometry.geom import Geometry
from geometry.sampler import Sampler
from network.net import Net
from network.activations import ActivationFactory, Sine
from physics.problems.cdr import CDR1D
from physics.problems.convection1D import convection1D
from physics.problems.helmholtz import helmholtz2D_annulus
from pinn import PINN
from training.trainer import Trainer
from validation.metrics import relative_l2
from validation.references import advection_diffusion_periodic, helmholtz_annulus


# --------------------------------------------------------------------------
# Problems. Each returns everything an arm needs, plus a grid of the swept
# parameter and the exact solution at every point of it.
# --------------------------------------------------------------------------

def _periodic_1d(x_max, n_bound=128):
    zero = lambda t, x: torch.zeros_like(x)
    return {
        "left":  {"p": [0., 1.], "x": lambda p: torch.zeros_like(p), "N": n_bound,
                  "periodic": True, "with": "right", "bc": {"u": {"type": "dirichlet", "value": zero}}},
        "right": {"p": [0., 1.], "x": lambda p, m=x_max: torch.full_like(p, m), "N": n_bound,
                  "periodic": True, "with": "left",  "bc": {"u": {"type": "dirichlet", "value": zero}}},
    }


def problem_cdr(b_max=5.0, nu=0.01, sigma=math.pi / 4):
    L = 2.0 * math.pi
    bounds = _periodic_1d(L)
    geo    = Geometry(bounds, dim=1, has_time=True, T=[0.0, 1.0])
    samp   = Sampler(geo, n_interior=2048, n_boundary=128, n_initial=128)

    TT, XX = torch.meshgrid(torch.linspace(0, 1, 21), torch.linspace(0, L, 96), indexing="ij")
    tg, xg = TT.reshape(-1), XX.reshape(-1)
    ic = lambda x: sum(torch.exp(-(x - math.pi + m * L) ** 2 / (2 * sigma ** 2))
                       for m in range(-4, 5))
    grid = torch.linspace(0.0, b_max, 11)

    def make_physics(device):
        ph = CDR1D(dim=1, has_time=True, device=device)
        ph.setParameters(
            params=[{"beta": b.item(), "nu": nu, "rho": 0.0} for b in torch.linspace(0, b_max, 4)],
            boundaries=bounds, initial={"u": ic},
            limits={"beta": {"min": 0.0, "max": b_max, "N": 4, "scale": "linear"}},
            hard_ic=True,
        )
        return ph

    def exact(b):
        return torch.as_tensor(advection_diffusion_periodic(tg, xg, b, nu, sigma=sigma),
                               dtype=torch.float32)

    mu = lambda g: torch.stack([g, torch.full_like(g, nu), torch.zeros_like(g)], 1)
    return dict(geo=geo, samp=samp, make_physics=make_physics, coords=torch.stack([tg, xg], 1),
                grid=grid, mu_of=mu, exact=exact, var="u", x_dim=2, mu_dim=3)


def problem_convection(b_max=5.0):
    L = 2.0 * math.pi
    bounds = _periodic_1d(L)
    geo    = Geometry(bounds, dim=1, has_time=True, T=[0.0, 1.0])
    samp   = Sampler(geo, n_interior=2048, n_boundary=128, n_initial=128)

    TT, XX = torch.meshgrid(torch.linspace(0, 1, 21), torch.linspace(0, L, 96), indexing="ij")
    tg, xg = TT.reshape(-1), XX.reshape(-1)
    grid = torch.linspace(0.5, b_max, 11)

    def make_physics(device):
        ph = convection1D(dim=1, has_time=True, device=device)
        ph.setParameters(
            params=[{"beta": b.item()} for b in torch.linspace(0.5, b_max, 4)],
            boundaries=bounds, initial={"u": lambda x: 1.0 + torch.sin(x)},
            limits={"beta": {"min": 0.5, "max": b_max, "N": 4, "scale": "linear"}},
        )
        return ph

    exact = lambda b: 1.0 + torch.sin(xg - b * tg)
    return dict(geo=geo, samp=samp, make_physics=make_physics, coords=torch.stack([tg, xg], 1),
                grid=grid, mu_of=lambda g: g.unsqueeze(1), exact=exact,
                var="u", x_dim=2, mu_dim=1)


def problem_helmholtz(k_min=1.0, k_max=12.0, R=3.0, r=1.0, n_arcs=8):
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
    samp = Sampler(geo, n_interior=4096, n_boundary=64)

    rho = torch.linspace(r, R, 40)
    th  = torch.linspace(0, 2 * math.pi, 81)[:-1]
    RHO, TH = torch.meshgrid(rho, th, indexing="ij")
    rx, ry  = (RHO * torch.cos(TH)).reshape(-1), (RHO * torch.sin(TH)).reshape(-1)
    grid = torch.linspace(k_min, k_max, 9)

    def make_physics(device):
        ph = helmholtz2D_annulus(dim=2, has_time=False, device=device)
        ph.setParameters(
            params=[{"k": k.item()} for k in torch.linspace(k_min, k_max, 4)],
            boundaries=bounds,
            limits={"k": {"min": k_min, "max": k_max, "N": 4, "scale": "linear"}},
        )
        return ph

    def exact(k):
        return torch.as_tensor(helmholtz_annulus(rx.numpy(), ry.numpy(), k, r, R, n_arcs),
                               dtype=torch.float32)

    # Geometry stores 2D coordinates as (y, x)
    return dict(geo=geo, samp=samp, make_physics=make_physics,
                coords=torch.stack([ry, rx], 1), grid=grid,
                mu_of=lambda g: g.unsqueeze(1), exact=exact,
                var="u", x_dim=2, mu_dim=1)


PROBLEMS = {"cdr": problem_cdr, "convection": problem_convection, "helmholtz": problem_helmholtz}


# --------------------------------------------------------------------------
# Sweeps. Each arm is a dict of overrides applied on top of BASE.
# --------------------------------------------------------------------------

BASE = dict(dx=32, dmu=32, d_h=64, encoder_layers=2, trunk_layers=3,
            head_layers=2, film_layers=2, use_film=True,
            optimiser="nadam", balancing="lra", param_every=100, lr=1e-3)

SWEEPS = {
    "arch": [(f"{k}={v}", {k: v}) for k, vals in [
        ("dx", [16, 32, 64]), ("dmu", [8, 32, 64]), ("d_h", [32, 64, 128]),
        ("encoder_layers", [2, 3]), ("trunk_layers", [2, 3, 5]), ("head_layers", [2, 3]),
    ] for v in vals],
    # film_layers stays >= 2 so the generator keeps its activation: at 1 it is a
    # bare Linear and the block is tested in its weakest possible form.
    "film": [("film on",  {"use_film": True,  "film_layers": 2}),
             ("film off", {"use_film": False}),
             ("film deep", {"use_film": True, "film_layers": 3}),
             ("dmu=8",    {"dmu": 8}), ("dmu=64", {"dmu": 64})],
    "optimiser": [("nadam lr=1e-3",   {"optimiser": "nadam", "lr": 1e-3}),
                  ("nadam lr=1e-4",   {"optimiser": "nadam", "lr": 1e-4}),
                  ("hypergrad",       {"optimiser": "hypergrad"}),
                  ("hypergrad lr0=1e-5", {"optimiser": "hypergrad", "lr": 1e-5})],
    "balancing": [("lra", {"balancing": "lra"}), ("none", {"balancing": "none"})],
    "sweeprate": [(f"param_every={v}", {"param_every": v}) for v in (0, 100, 500)],
}


_CACHE = {}


def get_problem(name):
    """
    Build a problem once per process and keep it.

    Both halves of the setup are expensive and neither depends on the arm or
    the seed: on the Helmholtz annulus one draw of the interior pool costs ~7 s
    (the ray-casting inside-test runs over sixteen boundary segments) and the
    analytic reference costs ~2 s per grid point. Rebuilding them for every arm
    dominated the run.
    """
    if name not in _CACHE:
        P = PROBLEMS[name]()
        P["exact_cache"] = [P["exact"](b) for b in P["grid"].tolist()]
        _CACHE[name] = P
    return _CACHE[name]


def run_arm(problem, cfg, seed, steps, device):
    P = get_problem(problem)
    torch.manual_seed(seed)
    net = Net(
        x_dim=P["x_dim"], mu_dim=P["mu_dim"],
        dx=cfg["dx"], dmu=cfg["dmu"], d_h=cfg["d_h"],
        encoder_layers=cfg["encoder_layers"], trunk_layers=cfg["trunk_layers"],
        head_layers=cfg["head_layers"], film_layers=cfg["film_layers"],
        activation=ActivationFactory(Sine, omega=1.0, trainable=True),
        encoder_activation=nn.Tanh, film_activation=nn.Tanh,
        outputs_config=({"u": {"activation": nn.Tanh}, "ux": {"activation": nn.Tanh}}
                        if problem == "cdr" else {"u": {"activation": nn.Tanh}}),
        use_film=cfg["use_film"], use_fourier=False,
    )
    ph = P["make_physics"](device)
    torch.manual_seed(seed + 10_000)
    pinn = PINN(net, ph, P["samp"], n_refine=1, adaptive_pde=True, device=device)
    tr = Trainer(pinn=pinn, lr=cfg["lr"], n_iter=steps, resample_every=1000,
                 checkpoint_every=10 ** 9, gradnorm_every=200, lra_alpha=0.99,
                 balancing=cfg["balancing"], optimiser=cfg["optimiser"],
                 param_every=cfg["param_every"], checkpoint_path="/tmp/bench",
                 run_name="bench", save_final=False, logger="none", device=device, progress=False,
                 log_every=max(1, steps // 10))
    t0 = time.time()
    tr.train()
    wall = time.time() - t0

    grid   = P["grid"]
    coords = P["coords"].to(device)
    with torch.no_grad():
        pred = pinn.predict(coords, P["mu_of"](grid).to(device))[P["var"]]
    profile = [relative_l2(pred[:, j].cpu(), ref)
               for j, ref in enumerate(P["exact_cache"])]
    return {
        "mean_l2": sum(profile) / len(profile),
        "profile": profile,
        "grid": grid.tolist(),
        "wall_s": round(wall, 1),
        "n_params": sum(p.numel() for p in net.parameters()),
        "final_lr": getattr(tr.optimiser, "last_eta", cfg["lr"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True, choices=sorted(PROBLEMS))
    ap.add_argument("--sweep",   required=True, choices=sorted(SWEEPS))
    ap.add_argument("--seeds",   type=int, default=16)
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="shift the seed range, so array jobs cover disjoint seeds")
    ap.add_argument("--steps",   type=int, default=20000)
    ap.add_argument("--device",  default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out",     default=None)
    args = ap.parse_args()

    out = pathlib.Path(args.out or f"results_{args.problem}_{args.sweep}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    arms  = SWEEPS[args.sweep]
    total = len(arms) * args.seeds
    stamp = lambda: time.strftime("%H:%M:%S")
    say   = lambda msg: print(f"[{stamp()}] {msg}", flush=True)

    say(f"{args.problem} / {args.sweep}   device {args.device}")
    say(f"{len(arms)} arms x {args.seeds} seeds = {total} runs, {args.steps} steps each")
    say(f"arms: {', '.join(a for a, _ in arms)}")
    say(f"writing to {out.resolve()}")

    t_start, done, failed, scores = time.time(), 0, 0, {}
    for label, over in arms:
        for s in range(args.seed_offset, args.seed_offset + args.seeds):
            done    += 1
            elapsed  = time.time() - t_start
            eta      = (elapsed / (done - 1)) * (total - done + 1) if done > 1 else 0.0
            say(f"run {done}/{total}  {label}  seed {s}"
                f"  (elapsed {elapsed/60:.1f}m, eta {eta/60:.0f}m)")

            cfg = {**BASE, **over}
            try:
                res = run_arm(args.problem, cfg, s, args.steps, args.device)
                rec = {"problem": args.problem, "sweep": args.sweep, "arm": label,
                       "seed": s, "steps": args.steps, **res}
                scores.setdefault(label, []).append(res["mean_l2"])
                say(f"  -> L2 {res['mean_l2']:.4f}   {res['wall_s']:.0f}s, "
                    f"{res['n_params']} weights, final lr {res['final_lr']:.2e}")
            except Exception as exc:                       # keep the array job alive
                failed += 1
                rec = {"problem": args.problem, "sweep": args.sweep, "arm": label,
                       "seed": s, "error": f"{type(exc).__name__}: {exc}"}
                say(f"  -> FAILED  {type(exc).__name__}: {exc}")

            with out.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")

        if label in scores:
            v = scores[label]
            say(f"== {label}: mean L2 {sum(v)/len(v):.4f} over {len(v)} seeds "
                f"(min {min(v):.4f}, max {max(v):.4f})")

    say(f"finished {total} runs in {(time.time()-t_start)/60:.1f}m, {failed} failed")
    if scores:
        say("ranking, best first:")
        for label, v in sorted(scores.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
            print(f"    {label:24s} {sum(v)/len(v):.4f}   "
                  f"seed spread x{max(v)/max(min(v), 1e-12):.2f}   n={len(v)}", flush=True)


if __name__ == "__main__":
    main()

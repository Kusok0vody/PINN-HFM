import sys
import math
import torch
import torch.nn as nn

sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))

from geometry.geom              import Geometry
from geometry.sampler           import Sampler
from network.net                import Net
from network.activations        import ActivationFactory, Sine
from physics.problems.helmholtz import helmholtz2D_annulus
from training.trainer           import Trainer
from pinn                       import PINN
from validation.validator       import Validator
from validation.references      import helmholtz_annulus, helmholtz_annulus_resonances

torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type != "cpu":
    torch.cuda.set_device(device)
print(f"Device: {device}")

# --- Geometry: annulus, u = 0 inside, alternating +/-1 over N_ARCS arcs outside ---
R      = 3.0
r      = 1.0
N_ARCS = 8

N_r   = 128
N_R   = 64
N_PDE = 4096

bounds = {}

for i in range(N_ARCS):
    p_start = i / N_ARCS
    p_end   = (i + 1) / N_ARCS
    bounds[f"rq{i+1}"] = {
        "p": [p_start, p_end],
        "x": lambda p, r=r: r * torch.cos(2 * math.pi * p),
        "y": lambda p, r=r: r * torch.sin(2 * math.pi * p),
        "N": N_r,
        "bc": {
            "u": {"type": "dirichlet", "value": lambda x, y: torch.zeros_like(x)},
        },
    }

for i in range(N_ARCS):
    p_start = i / N_ARCS
    p_end   = (i + 1) / N_ARCS
    value   = 1.0 if (i % 2 == 0) else -1.0
    bounds[f"Rq{i+1}"] = {
        "p": [p_start, p_end],
        "x": lambda p, R=R: R * torch.cos(2 * math.pi * p),
        "y": lambda p, R=R: R * torch.sin(2 * math.pi * p),
        "N": N_R,
        "bc": {
            "u": {"type": "dirichlet", "value": lambda x, y, val=value: val * torch.ones_like(x)},
        },
    }

geo  = Geometry(bounds, dim=2, has_time=False)
samp = Sampler(geo, n_interior=N_PDE, n_boundary=N_R)

pts = samp.sample()
print(f"interior: {pts.interior.coords.shape}")
for name, b in pts.boundaries.items():
    print(f"boundary '{name}': {b.coords.shape}")

# --- Parameter sweep, steering clear of the Dirichlet eigenvalues -------------
# The range stops below the first eigenvalue of this annulus at k = 6.513. It
# used to run to 12 with four sweep points, which happened to miss the
# resonance by 1.8; sixteen points over that range pass within 0.35 of it, and
# a solution that close has amplitude 9.8 against a median of 2.0, so its
# residual takes over the loss. Narrowing the range is what makes a denser
# sweep possible at all.
K_MIN = 1.0
K_MAX = 6.0
N_K   = 16

bad = helmholtz_annulus_resonances(K_MIN, K_MAX, r, R, N_ARCS)
ks  = torch.linspace(K_MIN, K_MAX, N_K)
if bad:
    nearest = min((ks - b).abs().min().item() for b in bad)
    print(f"resonant k in [{K_MIN}, {K_MAX}]: {[round(v, 3) for v in bad]}, "
          f"nearest sweep point {nearest:.3f} away")
    if nearest < 0.5:
        raise SystemExit(
            "A sweep point sits on a Dirichlet eigenvalue: the solution there is "
            "unbounded and its residual will dominate every other setting. "
            "Narrow K_MIN/K_MAX or reduce N_K."
        )
else:
    print(f"sweep k in [{K_MIN}, {K_MAX}] with {N_K} points, no resonances inside")

parameters = [{"k": k.item()} for k in ks]
limits     = {"k": {"min": K_MIN, "max": K_MAX, "N": N_K, "scale": "linear"}}

net = Net(
    x_dim=2, mu_dim=1,
    dx=32, dmu=64, d_h=64,
    encoder_layers=3,
    trunk_layers=4,
    head_layers=3,
    film_layers=2,
    activation=ActivationFactory(Sine, omega=1.0, trainable=False),
    encoder_activation=nn.Tanh,
    trunk_activation=ActivationFactory(Sine, omega=1.0, trainable=False),
    head_activation=nn.Tanh,
    film_activation=nn.Tanh,
    outputs_config={"u": {"activation": ActivationFactory(Sine, omega=1.0, trainable=True)}},
    use_film=True,
    use_fourier=False,
)

physics = helmholtz2D_annulus(dim=2, has_time=False, device=device)
physics.setParameters(
    params=parameters,
    boundaries=bounds,
    limits=limits,
)

pinn = PINN(
    net, physics, samp,
    n_refine=10,
    adaptive_pde=True,
    adaptive_bc=False,
    adaptive_ic=False,
    device=device,
)

# --- Analytic reference on a polar grid, one block per sweep point ------------
N_RHO, N_THETA = 64, 128
rho_grid = torch.linspace(r, R, N_RHO)
th_grid  = torch.linspace(0.0, 2 * math.pi, N_THETA + 1)[:-1]
RHO, TH  = torch.meshgrid(rho_grid, th_grid, indexing="ij")

ref_x = (RHO * torch.cos(TH)).reshape(-1)
ref_y = (RHO * torch.sin(TH)).reshape(-1)

# Geometry stores 2D coordinates as (y, x) — see Geometry.COORD_ORDER
ref_coords = torch.stack([ref_y, ref_x], dim=1)
ref_u = torch.stack(
    [
        torch.as_tensor(
            helmholtz_annulus(ref_x.numpy(), ref_y.numpy(), p["k"], r, R, N_ARCS),
            dtype=torch.float32,
        )
        for p in parameters
    ],
    dim=1,
)
amps = ref_u.abs().amax(dim=0)
print(f"reference block: {tuple(ref_u.shape)}  max|u| = {ref_u.abs().max():.3f}, "
      f"median over k = {amps.median():.3f}")
print(f"derivative path: {'paired' if pinn.paired_coords else 'broadcast'}, "
      f"M = {physics.par.tensor.shape[0]} parameter settings")

validator = Validator(
    pinn,
    ref_coords=ref_coords,
    ref_values={"u": ref_u},
    ref_params=physics.par.tensor.clone(),
    seed=0,
)
print(f"metrics before training: "
      f"{ {key: round(val, 5) for key, val in validator.evaluate().items()} }")

N_ITERS = 20000

trainer = Trainer(
    pinn=pinn,
    lr=1e-3,
    n_iter=N_ITERS,
    resample_every=1000,
    checkpoint_every=1000,
    gradnorm_every=200,
    lra_alpha=0.99,
    checkpoint_path="checkpoints",
    run_name="helmholtz",
    save_final=True,
    logger="tqdm",
    device=device,
    start_step=0,
    validator=validator,
    validate_every=200,
    log_every=N_ITERS // 40,
)

print(f"=== Training ({N_ITERS} iterations) ===")
trainer.train()

print("=== Final metrics ===")
for name, val in validator.evaluate().items():
    print(f"  {name}: {val:.6e}")

"""
Обучение PINN: уравнение Гельмгольца в кольце
    Delta u + k^2 * u = 0

Область: кольцо r=1 (внутр.) ... R=3 (внешн.)

Граничные условия:
  - Внутренняя граница (r=1): u = 0 (8 дуг, Дирихле)
  - Внешняя граница  (R=3): чередование u = +1 / -1 (8 секторов, Дирихле)

Параметр: k in {1, 2, 5, 10, 20, 50} — обучение на нескольких k одновременно.
Чекпоинты: checkpoints/helmholtz/ckpt_*.pt
"""

import sys
import math
import torch
import torch.nn as nn

sys.path.append("src/")

from geometry.geom                import Geometry
from geometry.sampler             import Sampler
from network.net                  import Net
from network.activations          import ActivationFactory, Sine
from physics.problems.Helmholtz import helmholtz2D_annulus
from training.trainer             import Trainer
from pinn                         import PINN

# ── Воспроизводимость ─────────────────────────────────────────────────────────
torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type != "cpu":
    torch.cuda.set_device(device)
print(f"Device: {device}")

# ── 1. Геометрия ──────────────────────────────────────────────────────────────
R = 3.0
r = 1.0

N_r   = 128   # точек на каждую из 8 внутренних дуг
N_R   = 64    # точек на каждую из 8 внешних дуг
N_PDE = 4096

bounds = {}

# Внутренняя граница (r = const) — 8 дуг, u = 0
for i in range(8):
    p_start = i / 8.0
    p_end   = (i + 1) / 8.0
    bounds[f"rq{i+1}"] = {
        "p": [p_start, p_end],
        "x": lambda p, r=r: r * torch.cos(2 * math.pi * p),
        "y": lambda p, r=r: r * torch.sin(2 * math.pi * p),
        "N": N_r,
        "bc": {
            "u": {"type": "dirichlet", "value": lambda x, y: torch.zeros_like(x)},
        },
    }

# Внешняя граница (R = const) — 8 дуг, чередование +1 / -1
for i in range(8):
    p_start = i / 8.0
    p_end   = (i + 1) / 8.0
    value = 1.0 if (i % 2 == 0) else -1.0
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

# ── 2. Сеть ───────────────────────────────────────────────────────────────────
net = Net(
    x_dim=2, mu_dim=1,
    dx=32, dmu=64, d_h=64,
    encoder_layers=3,
    trunk_layers=4,
    head_layers=3,
    activation=ActivationFactory(Sine, omega=1.0, trainable=False),
    encoder_activation=nn.Tanh,
    trunk_activation=ActivationFactory(Sine, omega=1.0, trainable=False),
    head_activation=nn.Tanh,
    film_activation=nn.Tanh,
    outputs_config={"u": {"activation": ActivationFactory(Sine, omega=1.0, trainable=True)}},
    use_film=True,
    use_fourier=False,
)

# ── 3. Физика ─────────────────────────────────────────────────────────────────
K_MIN = 1.0
K_MAX = 10.0
N_K   = 2

ks = torch.linspace(K_MIN, K_MAX, N_K)
parameters = [{"k": k.item()} for k in ks]
parameters = [{"k": 10}]
limits = {"k": {"min": K_MIN, "max": K_MAX, "N": N_K, "scale": "linear"}}

physics = helmholtz2D_annulus(dim=2, has_time=False, device=device)
physics.setParameters(
    params=parameters,
    boundaries=bounds,
    initial=None,
    # limits=limits,
)

# ── 4. PINN ───────────────────────────────────────────────────────────────────
pinn = PINN(
    net, physics, samp,
    n_refine=10,
    adaptive_pde=True,
    adaptive_bc=False,
    adaptive_ic=False,
    device=device,
)

# ── 5. Обучение ───────────────────────────────────────────────────────────────
N_ITERS = 20000

trainer = Trainer(
    pinn=pinn,
    lr=1e-3,
    n_iter=N_ITERS,
    resample_every=1000,
    checkpoint_every=100,
    gradnorm_every=200,
    lra_alpha=0.01,
    checkpoint_path="checkpoints",
    run_name="helmholtz",
    save_final=True,
    logger="tqdm",
    device=device,
    start_step=0,
)

print(f"=== Обучение ({N_ITERS} итераций) ===")
trainer.train()

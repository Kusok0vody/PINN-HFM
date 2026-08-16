"""
Обучение PINN: уравнение Лапласа в кольце
    Delta u = 0   (Гельмгольц с k = 0)

Область: кольцо r=1 (внутр.) ... R=3 (внешн.)

Граничные условия:
  Внутренняя граница (r=1), 4 квадранта:
    rq1 [0,   pi/2]: u = +1
    rq2 [pi/2, pi]:  u =  0
    rq3 [pi, 3pi/2]: u = +1
    rq4 [3pi/2, 2pi]:u =  0
  Внешняя граница (R=3), 4 квадранта:
    Rq1 [0,   pi/2]: u =  0
    Rq2 [pi/2, pi]:  u = -1
    Rq3 [pi, 3pi/2]: u =  0
    Rq4 [3pi/2, 2pi]:u = -1

Один набор параметров (k=0), статическая задача.
Чекпоинты: checkpoints/laplace/ckpt_*.pt
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

N_r   = 256
N_R   = 128
N_PDE = 4096

bounds = {
    "rq1": {
        "p": [0.0, 0.25],
        "x": lambda p: r * torch.cos(2 * math.pi * p),
        "y": lambda p: r * torch.sin(2 * math.pi * p),
        "N": N_r,
        "bc": {
            "u": {"type": "dirichlet", "value": lambda x, y: torch.ones_like(x)},
        },
    },
    "rq2": {
        "p": [0.25, 0.5],
        "x": lambda p: r * torch.cos(2 * math.pi * p),
        "y": lambda p: r * torch.sin(2 * math.pi * p),
        "N": N_r,
        "bc": {
            "u": {"type": "dirichlet", "value": lambda x, y: torch.zeros_like(x)},
        },
    },
    "rq3": {
        "p": [0.5, 0.75],
        "x": lambda p: r * torch.cos(2 * math.pi * p),
        "y": lambda p: r * torch.sin(2 * math.pi * p),
        "N": N_r,
        "bc": {
            "u": {"type": "dirichlet", "value": lambda x, y: torch.ones_like(x)},
        },
    },
    "rq4": {
        "p": [0.75, 1.0],
        "x": lambda p: r * torch.cos(2 * math.pi * p),
        "y": lambda p: r * torch.sin(2 * math.pi * p),
        "N": N_r,
        "bc": {
            "u": {"type": "dirichlet", "value": lambda x, y: torch.zeros_like(x)},
        },
    },
    "Rq1": {
        "p": [0.0, 0.25],
        "x": lambda p: R * torch.cos(2 * math.pi * p),
        "y": lambda p: R * torch.sin(2 * math.pi * p),
        "N": N_R,
        "bc": {
            "u": {"type": "dirichlet", "value": lambda x, y: torch.zeros_like(x)},
        },
    },
    "Rq2": {
        "p": [0.25, 0.5],
        "x": lambda p: R * torch.cos(2 * math.pi * p),
        "y": lambda p: R * torch.sin(2 * math.pi * p),
        "N": N_R,
        "bc": {
            "u": {"type": "dirichlet", "value": lambda x, y: -torch.ones_like(x)},
        },
    },
    "Rq3": {
        "p": [0.5, 0.75],
        "x": lambda p: R * torch.cos(2 * math.pi * p),
        "y": lambda p: R * torch.sin(2 * math.pi * p),
        "N": N_R,
        "bc": {
            "u": {"type": "dirichlet", "value": lambda x, y: torch.zeros_like(x)},
        },
    },
    "Rq4": {
        "p": [0.75, 1.0],
        "x": lambda p: R * torch.cos(2 * math.pi * p),
        "y": lambda p: R * torch.sin(2 * math.pi * p),
        "N": N_R,
        "bc": {
            "u": {"type": "dirichlet", "value": lambda x, y: -torch.ones_like(x)},
        },
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

# ── 3. Физика (k = 0 → уравнение Лапласа) ────────────────────────────────────
physics = helmholtz2D_annulus(dim=2, has_time=False, device=device)
physics.setParameters(
    params=[{"k": 0}],
    boundaries=bounds,
    initial=None,
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
    run_name="laplace",
    save_final=True,
    logger="tqdm",
    device=device,
    start_step=0,
)

print(f"=== Обучение ({N_ITERS} итераций) ===")
trainer.train()

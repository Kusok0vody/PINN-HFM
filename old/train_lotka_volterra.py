"""
Обучение PINN: трёхзвенная система Лотки-Вольтерры

    dG/dt = r*G*(1 - G/K) - alpha*G*H
    dH/dt = e1*alpha*G*H  - d1*H - beta*H*P
    dP/dt = e2*beta*H*P   - d2*P

Реализация: dim=1, has_time=False  (t — «x»-координата)
  - Левая граница (t=0): начальные условия G0, H0, P0 (Дирихле)
  - Правая граница (t=T): свободная (N=0, условий нет)

Чекпоинты: checkpoints/lotka_volterra/ckpt_*.pt
"""

import sys
import torch
import torch.nn as nn

sys.path.append("src/")

from geometry.geom                      import Geometry
from geometry.sampler                   import Sampler
from network.net                        import Net
from network.activations                import ActivationFactory, Sine
from physics.problems.lotka_volterra    import LotkaVolterra
from training.trainer                   import Trainer
from pinn                               import PINN

# ── Воспроизводимость ─────────────────────────────────────────────────────────
torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type != "cpu":
    torch.cuda.set_device(device)
print(f"Device: {device}")

# ── Параметры системы ─────────────────────────────────────────────────────────
T_END = 30.0

PARAMS = {
    "r":     1.0,
    "K":     1.0,
    "alpha": 1.0,
    "e1":    0.5,
    "d1":    0.3,
    "beta":  0.8,
    "e2":    0.6,
    "d2":    0.2,
}

N_PARAMS_BATCH = 10

params_definition = {
    "r":     {"min": 0.5,  "max": 2.0,  "N": N_PARAMS_BATCH, "scale": "linear"},
    "K":     {"min": 0.8,  "max": 1.2,  "N": N_PARAMS_BATCH, "scale": "linear"},
    "alpha": {"min": 0.2,  "max": 1.5,  "N": N_PARAMS_BATCH, "scale": "linear"},
    "e1":    {"min": 0.2,  "max": 0.8,  "N": N_PARAMS_BATCH, "scale": "linear"},
    "d1":    {"min": 0.1,  "max": 0.6,  "N": N_PARAMS_BATCH, "scale": "linear"},
    "beta":  {"min": 0.1,  "max": 1.2,  "N": N_PARAMS_BATCH, "scale": "linear"},
    "e2":    {"min": 0.2,  "max": 0.9,  "N": N_PARAMS_BATCH, "scale": "linear"},
    "d2":    {"min": 0.05, "max": 0.4,  "N": N_PARAMS_BATCH, "scale": "linear"},
}

# Начальные условия
G0 = 0.8
H0 = 0.4
P0 = 0.2

# ── 1. Геометрия ──────────────────────────────────────────────────────────────
N_IC  = 1    # точек для IC (левая граница)
N_PDE = 512   # внутренних точек

bounds = {
    "ic": {
        "p": [0.0, 1.0],
        "x": lambda p: torch.zeros_like(p),
        "N": N_IC,
        "bc": {
            "G": {"type": "dirichlet", "value": lambda x: G0 * torch.ones_like(x)},
            "H": {"type": "dirichlet", "value": lambda x: H0 * torch.ones_like(x)},
            "P": {"type": "dirichlet", "value": lambda x: P0 * torch.ones_like(x)},
        },
    },
    "right": {
        "p": [0.0, 1.0],
        "x": lambda p: T_END * torch.ones_like(p),
        "N": 0,
    },
}

geo  = Geometry(bounds, dim=1, has_time=False)
samp = Sampler(geo, n_interior=N_PDE, n_boundary=N_IC)

pts = samp.sample()
print(f"interior: {pts.interior.coords.shape}")
for name, b in pts.boundaries.items():
    print(f"boundary '{name}': {b.coords.shape}")

# ── 2. Сеть ───────────────────────────────────────────────────────────────────
net = Net(
    x_dim=1, mu_dim=8,
    dx=128, dmu=128, d_h=64,
    encoder_layers=4,
    trunk_layers=4,
    head_layers=3,
    film_layers=2,
    activation=nn.Tanh,
    encoder_activation=nn.ReLU,
    trunk_activation=nn.Tanh,
    head_activation=nn.Tanh,
    film_activation=nn.Tanh,
    outputs_config={
        "G": {},
        "H": {},
        "P": {},
    },
    use_film=True,
    use_fourier=False,
)

# ── 3. Физика ─────────────────────────────────────────────────────────────────
physics = LotkaVolterra(dim=1, has_time=False, device=device)
physics.setParameters(
    params=[PARAMS],
    boundaries=bounds,
    limits=params_definition,
    initial=None,
)

# ── 4. PINN ───────────────────────────────────────────────────────────────────
pinn = PINN(
    net, physics, samp,
    n_refine=1,
    adaptive_pde=True,
    adaptive_bc=False,
    adaptive_ic=False,
    device=device,
)

# ── 5. Обучение ───────────────────────────────────────────────────────────────
N_ITERS = 30000

trainer = Trainer(
    pinn=pinn,
    lr=1e-3,
    n_iter=N_ITERS,
    resample_every=25,
    checkpoint_every=1000,
    gradnorm_every=200,
    param_every=100,
    lra_alpha=0.01,
    checkpoint_path="checkpoints",
    run_name="lotka_volterra",
    save_final=False,
    logger="tqdm",
    device=device,
    start_step=0,
)

print(f"=== Обучение ({N_ITERS} итераций) ===")
trainer.train()

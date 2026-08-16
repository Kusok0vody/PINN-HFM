import sys
import math
import torch
import torch.nn as nn

sys.path.append("src/")

from geometry.geom               import Geometry
from geometry.sampler            import Sampler
from network.net                 import Net
from network.activations         import ActivationFactory, Sine
from physics.problems.elastic_wave import ElasticWave2D
from training.trainer            import Trainer
from pinn                        import PINN

# ── Воспроизводимость ─────────────────────────────────────────────────────────
torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type != "cpu":
    torch.cuda.set_device(device)
print(f"Device: {device}")

# ── 1. Геометрия (прямоугольник) ─────────────────────────────────────────────
X_MIN, X_MAX = 0.0, 30.0
Y_MIN, Y_MAX = 0.0, 30.0
T_MIN, T_MAX = 0.0, 0.1

N_BC  = 128
N_PDE = 8192
N_IC  = 4096

zero_t = lambda t, x, y: torch.zeros_like(x)

# Каждая сторона параметризована p ∈ [0, 1].
# Для прямоугольника `bound["x"]` и `bound["y"]` принимают p и возвращают координату.
bc_pml = {
    "vx": {"type": "dirichlet", "value": zero_t},
    "vy": {"type": "dirichlet", "value": zero_t},
}

bounds = {
    "bottom": {
        "p": [0.0, 1.0],
        "x": lambda p: X_MIN + (X_MAX - X_MIN) * p,
        "y": lambda p: 0 * p,
        "N": N_BC,
        # "bc": bc_pml,
    },
    "right": {
        "p": [0.0, 1.0],
        "x": lambda p: X_MAX * p,
        "y": lambda p: Y_MIN + (Y_MAX - Y_MIN) * p,
        "N": N_BC,
        # "bc": bc_pml,
    },
    "top": {
        "p": [0.0, 1.0],
        "x": lambda p: X_MAX - (X_MAX - X_MIN) * p,
        "y": lambda p: Y_MAX * p,
        "N": N_BC,
        # "bc": bc_pml,
    },
    "left": {
        "p": [0.0, 1.0],
        "x": lambda p: 0 * p,
        "y": lambda p: Y_MAX - (Y_MAX - Y_MIN) * p,
        "N": N_BC,
        # "bc": bc_pml,
    },
}

geo  = Geometry(bounds, dim=2, has_time=True, T=[T_MIN, T_MAX])
samp = Sampler(geo, n_interior=N_PDE, n_boundary=N_BC, n_initial=N_IC)

pts = samp.sample()
print(f"interior: {pts.interior.coords.shape}")
print(f"initial:  {pts.initial.coords.shape}")
for name, b in pts.boundaries.items():
    print(f"boundary '{name}': {b.coords.shape}")

# ── 2. Сеть ───────────────────────────────────────────────────────────────────
sine_t = ActivationFactory(Sine, omega=1.0, trainable=True)
sine_f = ActivationFactory(Sine, omega=1.0, trainable=True)

net = Net(
    x_dim=3, mu_dim=1,
    dx=64, dmu=32, d_h=128,
    encoder_layers=3,
    trunk_layers=5,
    head_layers=3,
    film_layers=2,
    activation=sine_f,
    encoder_activation=nn.Tanh,
    trunk_activation=sine_f,
    head_activation=nn.Tanh,
    film_activation=nn.Tanh,
    outputs_config={
        "vx":  {"activation": sine_t},
        "vy":  {"activation": sine_t},
        "sxx": {"activation": sine_t},
        "syy": {"activation": sine_t},
        "sxy": {"activation": sine_t},
    },
    use_film=False,
    use_fourier=False,
    n_freqs=16,
    omega_min=1.0,
    omega_max=64.0,
)

# ── 3. Физика ─────────────────────────────────────────────────────────────────
F0 = 1400/(500 * 0.15)
T0 = 1.0 / F0
H  = 0.15

medium = {
    "rho": lambda x, y: torch.ones_like(x) * 1.0,
    "Vp":  lambda x, y: torch.ones_like(x) * 1.0,
    "Vs":  lambda x, y: torch.ones_like(x) * 0.5,
}

source = {
    "xs": 7.5,
    "ys": 7.5,
    "t0": T0,
    "h":  H,
}

domain = {
    "x_min": X_MIN, "x_max": X_MAX,
    "y_min": Y_MIN, "y_max": Y_MAX,
}

physics = ElasticWave2D(dim=2, has_time=True, device=device)
physics.setParameters(
    params=[{"f0": F0}],
    boundaries=bounds,
    medium=medium,
    source=source,
    domain=domain,
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
print(pinn)

# ── 5. Обучение ───────────────────────────────────────────────────────────────
N_ITERS = 15000

trainer = Trainer(
    pinn=pinn,
    lr=1e-3,
    n_iter=N_ITERS,
    resample_every=1000,
    checkpoint_every=500,
    gradnorm_every=500,
    lra_alpha=0.01,
    checkpoint_path="checkpoints",
    run_name="elastic",
    save_final=True,
    logger="tqdm",
    device=device,
    start_step=0,
)

print(f"=== Обучение ({N_ITERS} итераций) ===")
trainer.train()

print("=== Residuals ===")
residuals = pinn.step()
for group, res_dict in residuals.items():
    group_total = 0
    for name, res in res_dict.items():
        w    = trainer.adaptive_weights.get(name, trainer.adaptive_weights.get(group, 1.0))
        val  = w * (res**2).mean().item()
        group_total += val
        print(f"  {group}/{name}: raw={res.abs().mean().item():.6f}  weighted={val:.6f}")
    print(f"  {group} total: {group_total:.6f}")
print(f"lr: {trainer.optimiser.param_groups[0]['lr']:.2e}")
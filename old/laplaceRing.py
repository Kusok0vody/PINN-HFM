import sys
import math
import torch
import torch.nn as nn

sys.path.append("src/")

from geometry.geom                 import Geometry
from geometry.sampler              import Sampler
from network.net                   import Net
from network.activations           import ActivationFactory, Sine, Morlet
from physics.problems.Helmholtz  import helmholtz2D_annulus
from visualization.plot            import plot_samples
from training.trainer              import Trainer
from pinn                          import PINN

torch.manual_seed(42)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device='cpu'
if device != 'cpu':
    torch.cuda.set_device(device)
print(device)

# # 1. Geometry
# R = 2
# r = 0.5

# N_r = 2048
# N_R = 1024
N_pde   = 4096

# bounds = {
#     "rq1": {
#         "p": [0.0, 0.25],
#         "x": lambda p: r * torch.cos(2 * math.pi * p),
#         "y": lambda p: r * torch.sin(2 * math.pi * p),
#         "N": N_r,
#         "bc": {
#             "u": {"type": "dirichlet", "value": lambda x, y: torch.ones_like(x)},
#         },
#     },
#     "rq2": {
#         "p": [0.25, 0.5],
#         "x": lambda p: r * torch.cos(2 * math.pi * p),
#         "y": lambda p: r * torch.sin(2 * math.pi * p),
#         "N": N_r,
#         "bc": {
#             "u": {"type": "dirichlet", "value": lambda x, y: torch.zeros_like(x)},
#         },
#     },
#     "rq3": {
#         "p": [0.5, 0.75],
#         "x": lambda p: r * torch.cos(2 * math.pi * p),
#         "y": lambda p: r * torch.sin(2 * math.pi * p),
#         "N": N_r,
#         "bc": {
#             "u": {"type": "dirichlet", "value": lambda x, y: torch.ones_like(x)},
#         },
#     },
#     "rq4": {
#         "p": [0.75, 1.0],
#         "x": lambda p: r * torch.cos(2 * math.pi * p),
#         "y": lambda p: r * torch.sin(2 * math.pi * p),
#         "N": N_r,
#         "bc": {
#             "u": {"type": "dirichlet", "value": lambda x, y: torch.zeros_like(x)},
#         },
#     },
#     "Rq1": {
#         "p": [0.0, 0.25],
#         "x": lambda p: R * torch.cos(2 * math.pi * p),
#         "y": lambda p: R * torch.sin(2 * math.pi * p),
#         "N": N_R,
#         "bc": {
#             "u": {"type": "dirichlet", "value": lambda x, y: torch.zeros_like(x)},
#         },
#     },
#     "Rq2": {
#         "p": [0.25, 0.5],
#         "x": lambda p: R * torch.cos(2 * math.pi * p),
#         "y": lambda p: R * torch.sin(2 * math.pi * p),
#         "N": N_R,
#         "bc": {
#             "u": {"type": "dirichlet", "value": lambda x, y: -torch.ones_like(x)},
#         },
#     },
#     "Rq3": {
#         "p": [0.5, 0.75],
#         "x": lambda p: R* torch.cos(2 * math.pi * p),
#         "y": lambda p: R * torch.sin(2 * math.pi * p),
#         "N": N_R,
#         "bc": {
#             "u": {"type": "dirichlet", "value": lambda x, y: torch.zeros_like(x)},
#         },
#     },
#     "Rq4": {
#         "p": [0.75, 1.0],
#         "x": lambda p: R * torch.cos(2 * math.pi * p),
#         "y": lambda p: R * torch.sin(2 * math.pi * p),
#         "N": N_R,
#         "bc": {
#             "u": {"type": "dirichlet", "value": lambda x, y: -torch.ones_like(x)},
#         },
#     },
# }

# bounds = {
#     "inlet": {
#         "p": [0.0, 1.0],
#         "x": lambda p: r * torch.cos(2 * math.pi * p),
#         "y": lambda p: r * torch.sin(2 * math.pi * p),
#         "N": N_inlet,
#         "bc": {
#             "u": {"type": "dirichlet", "value": lambda x, y: torch.ones_like(x)},
#         },
#     },
#     "outlet": {
#         "p": [0.0, 1.0],
#         "x": lambda p: R * torch.cos(2 * math.pi * p),
#         "y": lambda p: R * torch.sin(2 * math.pi * p),
#         "N": N_out,
#         "bc": {
#             "u": {"type": "dirichlet", "value": lambda x, y: -torch.ones_like(x)},
#         },
#     },
# }

# 1. Geometry
R = 3.0      # внешний радиус
r = 1.0      # внутренний радиус

N_r = 1024   # точек на внутренней дуге (на все 8 дуг)
N_R = 512   # точек на внешней дуге (на все 8 дуг)

bounds = {}

# Внутренняя граница (r = const) – 8 дуг, везде u = 0
# for i in range(8):
#     p_start = i / 8.0
#     p_end   = (i + 1) / 8.0
#     bounds[f"rq{i+1}"] = {
#         "p": [p_start, p_end],
#         "x": lambda p, r=r: r * torch.cos(2 * math.pi * p),
#         "y": lambda p, r=r: r * torch.sin(2 * math.pi * p),
#         "N": N_r,
#         "bc": {
#             "u": {"type": "dirichlet", "value": lambda x, y: torch.zeros_like(x)},
#         },
#     }

# Внешняя граница (R = const) – 8 дуг, чередование +1 / -1
for i in range(8):
    p_start = i / 8.0
    p_end   = (i + 1) / 8.0
    # чётные дуги (i=0,2,4,6) -> +1, нечётные -> -1
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
samp = Sampler(geo, n_interior=N_pde, n_boundary=N_R)

pts = samp.sample()
print("=== Geometry & Sampler ===")
print(f"  interior: {pts.interior.coords.shape}")
for name, b in pts.boundaries.items():
    print(f"  boundary '{name}': {b.coords.shape}")
print()

# import matplotlib.pyplot as plt
# plot_samples(pts, title="Test geometry")
# plt.savefig("ring_geometry.png", bbox_inches="tight", dpi=150)

# # 2. Network
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
    outputs_config={
        "u": {"activation": ActivationFactory(Sine, omega=1.0, trainable=True)},
    },
    use_film=True,
    use_fourier=False,
)

X  = torch.randn(10, 2)
mu = torch.randn(3, 1)
out = net(X, mu)
for name, val in out.items():
    print(f"  out['{name}']: {val.shape}")
print()

# 3. Physics
k_min = 1.0
k_max = 50.0
N_k   = 5

ks = torch.linspace(k_min, k_max, N_k)

parameters = [
    {"k": 1}
    # for k in ks
]

limits = {
    "k": {"min": k_min, "max": k_max, "N": N_k, "scale": "linear"}
}

print(parameters)

physics = helmholtz2D_annulus(dim=2, has_time=False, device=device)

physics.setParameters(
    params=parameters,
    boundaries=bounds,
    initial=None,
    # limits=limits,
)

# 4. PINN
pinn = PINN(
    net, physics, samp,
    n_refine=10,
    adaptive_pde=True,
    adaptive_bc=False,
    adaptive_ic=False,
    device=device
)

# residuals = pinn.step()
# print("=== Residuals ===")
# for group, res_dict in residuals.items():
#     for name, res in res_dict.items():
#         print(f"  {group}/{name}: {res.shape}  mean={res.abs().mean().item():.4f}")
# print()
# print("Points shape: ", pinn.points.interior.coords.shape)

# 5. Trainer
n_iters = 20000
start   = 0

trainer = Trainer(
    pinn=pinn,
    lr=1e-3,
    n_iter=n_iters,
    resample_every=1000,
    checkpoint_every=1000,
    gradnorm_every=200,
    # param_every=500,
    lra_alpha=0.01,
    checkpoint_path="checkpoints",
    run_name="helmholtz",
    save_final=False,
    logger="tqdm",
    device=device,
    start_step = start,
)

# Trainer.load_checkpoint(
#     path="checkpoints/proppant_debug/ckpt_start.pt",
#     pinn=pinn,
#     optimiser=trainer.optimiser,
#     scheduler=trainer.scheduler,
#     device=device,
# )

print(f"=== Training ({n_iters} iterations) ===")
trainer.train()
print()

residuals = pinn.step()
print("=== Residuals ===")
for group, res_dict in residuals.items():
    for name, res in res_dict.items():
        print(f"  {group}/{name}: {res.shape}  mean={res.abs().mean().item():.4f}")
print()

import matplotlib.pyplot as plt
pts = trainer.pinn.points
pts = pts.to("cpu")
plot_samples(pts, title="Test geometry")
plt.savefig("ring_geometry.png", bbox_inches="tight", dpi=150)

print("Points shape: ", pinn.points.interior.coords.shape)
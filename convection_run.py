import sys
import math
import torch
import torch.nn as nn

sys.path.append("src/")

from geometry.geom                 import Geometry
from geometry.sampler              import Sampler
from network.net                   import Net
from network.activations           import ActivationFactory, Sine, Morlet
from physics.problems.convection1D import convection1D
from visualization.plot            import plot_samples
from training.trainer              import Trainer
from pinn                          import PINN

torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device='cpu'
if device != 'cpu':
    torch.cuda.set_device(device)
print(device)

# 1. Geometry
N_bound = 256
N_pde   = 512
N_ic    = 256

bounds = {
    "left": {
        "p": [0, 1.0],
        "x": lambda p: 0*p,
        "N": N_bound,
        "bc": {
            "u": {"type": "dirichlet", "value": lambda t, x: torch.zeros_like(x)},
        },
        "periodic": True,
        "with": "right"
    },
    "right": {
        "p": [0.0, 1.0],
        "x": lambda p: 0*p + 6,
        "N": N_bound,
        "bc": {
            "u": {"type": "dirichlet", "value": lambda t, x: torch.zeros_like(x)},
        "periodic": True,
        "with": "left"
        }
    },
}

geo  = Geometry(bounds, dim=1, has_time=True, T=[0.0, 1.0])
samp = Sampler(geo, n_interior=N_pde, n_boundary=N_bound, n_initial=N_ic)

pts = samp.sample()
print("=== Geometry & Sampler ===")
print(f"  interior: {pts.interior.coords.shape}")
print(f"  initial:  {pts.initial.coords.shape}")
for name, b in pts.boundaries.items():
    print(f"  boundary '{name}': {b.coords.shape}")
print()

import matplotlib.pyplot as plt
plot_samples(pts, title="Test geometry")
plt.savefig("convection_geometry.png", bbox_inches="tight", dpi=150)

# 2. Network
net = Net(
    x_dim=2, mu_dim=1,
    dx=64, dmu=32, d_h=64,
    encoder_layers=2,
    trunk_layers=3,
    head_layers=3,
    activation=nn.Tanh,
    encoder_activation=nn.Tanh,
    trunk_activation=ActivationFactory(Sine, omega=0.5, trainable=True),
    head_activation=nn.Tanh,
    film_activation=nn.Tanh,
    outputs_config={
        "u":  {"activation": ActivationFactory(Morlet, omega=3.0, trainable=True)},
    },
    use_film=False,
    use_fourier=False,
)

print("=== Network ===")
print(net)
print()

X  = torch.randn(10, 2)
mu = torch.randn(3, 1)
out = net(X, mu)
for name, val in out.items():
    print(f"  out['{name}']: {val.shape}")
print()

# 3. Physics
beta_min = 0.01
beta_mean = 5
beta_max = 100
N_beta = 10

parameters = [
    {"beta": beta_mean},
]

print(parameters)

physics = convection1D(dim=1, has_time=True, device=device)
physics.setParameters(
    params=parameters,
    boundaries=bounds,
    initial={
        "u": lambda x: torch.ones_like(x) + torch.sin(x),
    }
)

print("=== Physics ===")
print(f"  boundaries:  {list(physics.boundaries.keys())}")
print()

# 4. PINN
pinn = PINN(
    net, physics, samp,
    n_refine=10,
    adaptive_pde=True,
    adaptive_bc=True,
    adaptive_ic=True,
    device=device
)

print("=== PINN ===")
print(pinn)
print()

residuals = pinn.step()
print("=== Residuals ===")
for group, res_dict in residuals.items():
    for name, res in res_dict.items():
        print(f"  {group}/{name}: {res.shape}  mean={res.abs().mean().item():.4f}")
print()
print("Points shape: ", pinn.points.interior.coords.shape)

# 5. Trainer
n_iters = 5000
start   = 0

trainer = Trainer(
    pinn=pinn,
    lr=1e-3,
    n_iter=n_iters,
    resample_every=2000,
    checkpoint_every=1000,
    checkpoint_path="checkpoints",
    run_name="proppant_debug",
    save_final=False,
    logger="tensorboard",
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
import sys
import math
import torch
import torch.nn as nn

sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))

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
if device != 'cpu':
    torch.cuda.set_device(device)
print(device)

N_bound = 1024
N_pde   = 2048
N_ic    = 2048

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
        "x": lambda p: 0*p + 2*torch.pi,
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

net = Net(
    x_dim=2, mu_dim=1,
    dx=32, dmu=64, d_h=64,
    encoder_layers=3,
    film_layers=1,
    trunk_layers=4,
    head_layers=3,
    activation=ActivationFactory(Sine, omega=1.0, trainable=True),
    encoder_activation=nn.Tanh,
    trunk_activation=ActivationFactory(Sine, omega=1.0, trainable=True),
    head_activation=ActivationFactory(Sine, omega=1.0, trainable=True),
    film_activation=nn.Tanh,
    outputs_config={
        "u": {"activation": nn.Tanh},
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

beta_min = 1.0
beta_max = 20.0
N_beta = 10

betas = torch.linspace(beta_min, beta_max, N_beta)

parameters = [
    # {"beta": b.item()}
    # for b in betas
    {"beta": 5.0}
]

limits = {
    "beta": {"min": beta_min, "max": beta_max, "N": N_beta, "scale": "linear"}
}

print(parameters)

physics = convection1D(dim=1, has_time=True, device=device)
physics.setParameters(
    params=parameters,
    boundaries=bounds,
    initial={
        "u": lambda x: torch.ones_like(x) + torch.sin(x),
    },
    # limits=limits,
)

pinn = PINN(
    net, physics, samp,
    n_refine=10,
    adaptive_pde=True,
    adaptive_bc=False,
    adaptive_ic=False,
    device=device
)


residuals = pinn.step()
print("=== Residuals ===")
for group, res_dict in residuals.items():
    for name, res in res_dict.items():
        print(f"  {group}/{name}: {res.shape}  mean={res.abs().mean().item():.4f}")
print()
print("Points shape: ", pinn.points.interior.coords.shape)

n_iters = 20000
start   = 0

trainer = Trainer(
    pinn=pinn,
    lr=1e-3,
    n_iter=n_iters,
    resample_every=1000,
    checkpoint_every=100,
    gradnorm_every=200,
    param_every=500,
    lra_alpha=0.99,
    checkpoint_path="checkpoints",
    run_name="convection",
    save_final=False,
    logger="tqdm",
    device=device,
    start_step = start,
)


print(f"=== Training ({n_iters} iterations) ===")
trainer.train()
print()

residuals = pinn.step()
print("=== Residuals ===")
for group, res_dict in residuals.items():
    for name, res in res_dict.items():
        print(f"  {group}/{name}: {res.shape}  mean={res.abs().mean().item():.4f}")
print()
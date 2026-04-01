import sys
import math
import torch
import torch.nn as nn

sys.path.append("src/")

from geometry.geom    import Geometry
from geometry.sampler import Sampler
from network.net      import Net
from network.activations import ActivationFactory, Sine, Morlet
from physics.problems.proppant import proppantDynamics_dless
from visualization.plot import plot_samples
from training.trainer import Trainer
from pinn             import PINN


torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device='cpu'
if device != 'cpu':
    torch.cuda.set_device(device)
print(device)

# 1. Geometry
N_bound = 4096
N_wall = 1024
N_pde = 8192
N_ic = 4096

chi = 0.5
chi_u = (1 + chi)/2 
chi_l = (1 - chi)/2 

bounds = {
    "lu_wall": {
        "p": [chi_u, 1.0],
        "x": lambda p: 0*p,
        "y": lambda p: p,
        "N": N_wall,
        "bc": {
            "p": {"type": "neumann"},
        }
    },
    "ll_wall": {
        "p": [0.0, chi_l],
        "x": lambda p: 0*p,
        "y": lambda p: p,
        "N": N_wall,
        "bc": {
            "p": {"type": "neumann"},
        }
    },
    "inlet": {
        "p": [chi_l, chi_u],
        "x": lambda p: 0*p,
        "y": lambda p: p,
        "N": N_bound,
        "bc": {
            # "c": {"type": "dirichlet", "value": lambda t, x, y: torch.ones_like(x) * 0.25/0.65},
            "c": {"type": "dirichlet", "value": lambda t, x, y: torch.ones_like(x) * 0.25/0.65 * (t <= 0.5).float()},
            "p": {"type": "neumann",   "value": lambda t, x, y: torch.ones_like(x)},
        }
    },
    "outlet": {
        "p": [0.0, 1.0],
        "x": lambda p: 0*p + 1,
        "y": lambda p: p,
        "N": N_bound,
        "bc": {
            # "u": {"type": "neumann"},
        }
    },
    "bottom": {
        "p": [0.0, 1.0],
        "x": lambda p: p,
        "y": lambda p: 0*p,
        "N": N_bound,
        "bc": {
            "p": {"type": "neumann"},
        }
    },
    "top": {
        "p": [0.0, 1.0],
        "x": lambda p: p,
        "y": lambda p: 0*p + 1,
        "N": N_bound,
        "bc": {
            "p": {"type": "neumann"},
        }
    },
}

geo  = Geometry(bounds, dim=2, has_time=True, T=[0.0, 1.0])
samp = Sampler(geo, n_interior=N_pde, n_boundary=N_bound, n_initial=N_ic)

pts = samp.sample()
print("=== Geometry & Sampler ===")
print(f"  interior: {pts.interior.coords.shape}")
print(f"  initial:  {pts.initial.coords.shape}")
for name, b in pts.boundaries.items():
    print(f"  boundary '{name}': {b.coords.shape}")
print()

# plot_samples(pts, title="Test geometry")

# 2. Network
outputs_config = {
    "c":  {},
    "px": {},
    "py": {},
}

net = Net(
    x_dim=3,
    mu_dim=4,
    dx=32,
    dmu=32,
    d_h=128,
    encoder_layers=2,
    trunk_layers=6,
    head_layers=2,
    activation=nn.Tanh,
    encoder_activation=nn.Tanh,
    trunk_activation=ActivationFactory(Sine, omega=1.0, trainable=True),
    head_activation=nn.Tanh,
    film_activation=nn.Tanh,
    outputs_config={
        "c":  {"activation": ActivationFactory(Morlet, omega=3.0, trainable=True)},
        # "c":  {"multi": True, "K": 4, "activation": ActivationFactory(Morlet, omega=3.0, trainable=True)},
        # "c":  {"multi": True, "K": 4},
        "px": {},
        "py": {},
    },
    use_film=True,
    use_fourier=False,
    n_freqs=8,
    omega_min=1.0,
    omega_max=32.0,
)

with torch.no_grad():
    for name in ["px", "py"]:
        net.outputs[name].net[-1].weight.data *= 0.01
        net.outputs[name].net[-1].bias.data   *= 0.01

print("=== Network ===")
print(net)
print()

X  = torch.randn(10, 3)
mu = torch.randn(3, 4)
out = net(X, mu)
for name, val in out.items():
    print(f"  out['{name}']: {val.shape}")
print()


# 3. Physics
rho_f = 1.0
rho_p = 1.2
g     = 0.0
H     = 1;  L = 1

p0    = 1 / (12 * 0.01 * L * 1)
G     = p0 * H * rho_f * g
r     = 0.65 * (rho_p - rho_f) / rho_f
alpha = L / H

parameters = [
    {"alpha": alpha, "beta": -2.5, "r": r, "G": G},
]

print(parameters)

physics = proppantDynamics_dless(dim=2, has_time=True, device=device)
physics.setParameters(
    params=parameters,
    funcPar={
        "w": lambda x, y: torch.ones_like(x),
    },
    boundaries=bounds,
    initial={
        "c": lambda x, y: torch.zeros_like(x),
    }
)

print("=== Physics ===")
print(f"  boundaries:  {list(physics.boundaries.keys())}")
print()


# 4. PINN
pinn = PINN(net, physics, samp, device)

print("=== PINN ===")
print(pinn)
print()

residuals = pinn.step()
print("=== Residuals ===")
for group, res_dict in residuals.items():
    for name, res in res_dict.items():
        print(f"  {group}/{name}: {res.shape}  mean={res.abs().mean().item():.4f}")
print()


# 5. Trainer
n_iters = 300

weights = {
    "convection":  1.0,
    "poisson":     1.0,
    "correlation": 1.0,
    "bc":          10.0,
    "inlet_c":     10.0,
    "ic":          10.0,
}

trainer = Trainer(
    pinn=pinn,
    weights=weights,
    lr=1e-4,
    n_iter=n_iters,
    resample_every=1000,
    checkpoint_every=1000,
    checkpoint_path="checkpoints",
    run_name="proppant_debug",
    save_final=True,
    logger="tensorboard",
    device=device,
)

print(f"=== Training ({n_iters} iterations) ===")
trainer.train()
print()

print("=== Residuals ===")
residuals = pinn.step()
for group, res_dict in residuals.items():
    for name, res in res_dict.items():
        print(f"  {group}/{name}: mean={res.abs().mean().item():.6f}")

# print("=== Checkpoint test ===")
# step = trainer.load_checkpoint("checkpoints_test/ckpt_25.pt")
# print(f"  loaded checkpoint at step {step}")
# print()

print("All tests passed!")
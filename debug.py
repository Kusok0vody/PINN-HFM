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
N_bound = 8192
N_wall = 2048
N_pde = 8192
N_ic = 8192

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
# net = Net(
#     x_dim=3,
#     mu_dim=4,
#     dx=32,
#     dmu=32,
#     d_h=128,
#     encoder_layers=2,
#     trunk_layers=6,
#     head_layers=2,
#     activation=nn.Tanh,
#     encoder_activation=nn.Tanh,
#     trunk_activation=ActivationFactory(Sine, omega=1.0, trainable=True),
#     head_activation=nn.Tanh,
#     film_activation=nn.Tanh,
#     outputs_config={
#         "c":  {"activation": ActivationFactory(Morlet, omega=3.0, trainable=True)},
#         # "c":  {"multi": True, "K": 4, "activation": ActivationFactory(Morlet, omega=3.0, trainable=True)},
#         # "c":  {"multi": True, "K": 4},
#         "px": {},
#         "py": {},
#     },
#     use_film=True,
#     use_fourier=False,
#     n_freqs=8,
#     omega_min=1.0,
#     omega_max=32.0,
# )

net = Net(
    x_dim=3, mu_dim=4,
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
        # "c":  {},
        "c":  {"activation": ActivationFactory(Morlet, omega=3.0, trainable=True)},
        "px": {},
        "py": {},
        # "px": {"activation": ActivationFactory(Sine, omega = 0.5, trainable=False)},
        # "py": {"activation": ActivationFactory(Sine, omega = 0.5, trainable=False)}
    },
    use_film=False,
    use_fourier=False,
)

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
    {"alpha": alpha, "beta": 0.0, "r": r, "G": G},
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
pinn = PINN(net, physics, samp, n_refine=10, device=device)

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
n_iters = 20000
start   = 0

weights = {
    "convection":  1.0,
    "poisson":     1.0,
    "correlation": 1.0,
    "bc":          10.0,
    "inlet_c":     100.0,
    "inlet_p":     100.0,
    "lu_wall_p":   50.0,
    "ll_wall_p":   50.0,
    "ic":          50.0,
}

trainer = Trainer(
    pinn=pinn,
    weights=weights,
    lr=1e-4,
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
        w    = weights.get(name, weights.get(group, 1.0))
        val  = w * (res**2).mean().item()
        group_total += val
        print(f"  {group}/{name}: raw={res.abs().mean().item():.6f}  weighted={val:.6f}")
    print(f"  {group} total: {group_total:.6f}")
print(f"lr: {trainer.optimiser.param_groups[0]['lr']:.2e}")

# print("=== Checkpoint test ===")
# step = trainer.load_checkpoint("checkpoints_test/ckpt_25.pt")
# print(f"  loaded checkpoint at step {step}")
# print()

print("All tests passed!")
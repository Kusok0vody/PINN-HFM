import sys
import math
import torch
import torch.nn as nn

sys.path.append("src/")

from geometry.geom    import Geometry
from geometry.sampler import Sampler
from network.net      import Net
from network.activations import Sine
from physics.problems.proppant import proppantDynamics_dless
from visualization.plot import plot_samples
from training.trainer import Trainer
from pinn             import PINN
from utils            import smooth_clamp


# =============================================================================
# 1. Geometry
# =============================================================================

bounds = {
    "inlet": {
        "p": [0.0, 1.0],
        "x": lambda p: 0*p,
        "y": lambda p: p,
        "N": 100,
        "bc": {
            "c": {"type": "dirichlet", "value": lambda t, x, y: torch.ones_like(x) * 0.3},
            "p": {"type": "neumann",   "value": lambda t, x, y: torch.ones_like(x)},
        }
    },
    "outlet": {
        "p": [0.0, 1.0],
        "x": lambda p: 0*p + 1,
        "y": lambda p: p,
        "N": 100,
        "bc": {
            "p": {"type": "neumann"},
        }
    },
    "bottom": {
        "p": [0.0, 1.0],
        "x": lambda p: p,
        "y": lambda p: 0*p,
        "N": 100,
        "bc": {
            "p": {"type": "neumann"},
        }
    },
    "top": {
        "p": [0.0, 1.0],
        "x": lambda p: p,
        "y": lambda p: 0*p + 1,
        "N": 100,
        "bc": {
            "p": {"type": "neumann"},
        }
    },
}

geo  = Geometry(bounds, dim=2, has_time=True, T=[0.0, 1.0])
samp = Sampler(geo, n_interior=500, n_boundary=100, n_initial=200)

pts = samp.sample()
print("=== Geometry & Sampler ===")
print(f"  interior: {pts.interior.coords.shape}")
print(f"  initial:  {pts.initial.coords.shape}")
for name, b in pts.boundaries.items():
    print(f"  boundary '{name}': {b.coords.shape}")
print()

# plot_samples(pts, title="Test geometry")


# =============================================================================
# 2. Network
# =============================================================================

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
    d_h=64,
    encoder_layers=2,
    trunk_layers=4,
    head_layers=2,
    activation=nn.Tanh,
    trunk_activation=Sine,
    outputs_config=outputs_config,
    use_film=True,
    use_fourier=True,
    n_freqs=8,
    omega_min=1.0,
    omega_max=32.0,
)

print("=== Network ===")
print(net)
print()

# smoke test
X  = torch.randn(10, 3)
mu = torch.randn(3, 4)
out = net(X, mu)
for name, val in out.items():
    print(f"  out['{name}']: {val.shape}")
print()


# =============================================================================
# 3. Physics
# =============================================================================

mu_list = [
    {"alpha": 1.0, "beta": 2.5, "r": 0.1, "G": 0.05},
    {"alpha": 1.5, "beta": 2.5, "r": 0.1, "G": 0.05},
    {"alpha": 2.0, "beta": 2.5, "r": 0.1, "G": 0.05},
]

print({
        "alpha": torch.tensor([m["alpha"] for m in mu_list]),
        "beta":  torch.tensor([m["beta"]  for m in mu_list]),
        "r":     torch.tensor([m["r"]     for m in mu_list]),
        "G":     torch.tensor([m["G"]     for m in mu_list]),
    })

physics = proppantDynamics_dless(dim=2, has_time=True)
physics.setParameters(
    params={
        "alpha": torch.tensor([m["alpha"] for m in mu_list]),
        "beta":  torch.tensor([m["beta"]  for m in mu_list]),
        "r":     torch.tensor([m["r"]     for m in mu_list]),
        "G":     torch.tensor([m["G"]     for m in mu_list]),
    },
    funcPar={
        "w": lambda x, y: torch.ones_like(x),
    },
    boundaries=bounds,
    initial={
        "c": lambda x, y: torch.zeros_like(x),
    }
)

print("=== Physics ===")
# print(f"  param_order: {physics.param_order}")
print(f"  boundaries:  {list(physics.boundaries.keys())}")
print()


# =============================================================================
# 4. PINN
# =============================================================================

pinn = PINN(net, physics, samp)

print("=== PINN ===")
print(pinn)
print()

mu_tensor = physics.set_par()
print(f"  mu tensor: {mu_tensor.shape}")
print()

residuals = pinn.step(mu_tensor)
print("=== Residuals ===")
for group, res_dict in residuals.items():
    for name, res in res_dict.items():
        print(f"  {group}/{name}: {res.shape}  mean={res.abs().mean().item():.4f}")
print()


# =============================================================================
# 5. Trainer — короткий тест
# =============================================================================

weights = {
    "pde":         1.0,
    "convection":  1.0,
    "poisson":     1.0,
    "correlation": 1.0,
    "bc":          10.0,
    "ic":          5.0,
}

trainer = Trainer(
    pinn=pinn,
    weights=weights,
    lr=1e-3,
    n_iter=50,
    resample_every=25,
    checkpoint_every=25,
    checkpoint_path="checkpoints_test",
    logger="tqdm",
    device="cpu",
)

print("=== Training (50 iterations) ===")
trainer.train()
print()

print("=== Checkpoint test ===")
step = trainer.load_checkpoint("checkpoints_test/ckpt_25.pt")
print(f"  loaded checkpoint at step {step}")
print()

print("All tests passed!")
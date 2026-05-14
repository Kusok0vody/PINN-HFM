import sys
import math
import torch
import torch.nn as nn

sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))

from geometry.geom                 import Geometry
from geometry.sampler              import Sampler
from network.net                   import Net
from network.activations           import ActivationFactory, Sine, Morlet
from physics.problems.helmholtz  import helmholtz2D_annulus
from visualization.plot            import plot_samples
from training.trainer              import Trainer
from pinn                          import PINN

torch.manual_seed(42)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
if device != 'cpu':
    torch.cuda.set_device(device)
print(device)


N_pde   = 4096

R = 3.0
r = 1.0

N_r = 1024
N_R = 512

bounds = {}


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
samp = Sampler(geo, n_interior=N_pde, n_boundary=N_R)

pts = samp.sample()
print("=== Geometry & Sampler ===")
print(f"  interior: {pts.interior.coords.shape}")
for name, b in pts.boundaries.items():
    print(f"  boundary '{name}': {b.coords.shape}")
print()


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

k_min = 1.0
k_max = 50.0
N_k   = 5

ks = torch.linspace(k_min, k_max, N_k)

parameters = [
    {"k": 1}
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
)

pinn = PINN(
    net, physics, samp,
    n_refine=10,
    adaptive_pde=True,
    adaptive_bc=False,
    adaptive_ic=False,
    device=device
)


n_iters = 20000
start   = 0

trainer = Trainer(
    pinn=pinn,
    lr=1e-3,
    n_iter=n_iters,
    resample_every=1000,
    checkpoint_every=1000,
    gradnorm_every=200,
    lra_alpha=0.01,
    checkpoint_path="checkpoints",
    run_name="helmholtz",
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

import matplotlib.pyplot as plt
pts = trainer.pinn.points
pts = pts.to("cpu")
plot_samples(pts, title="Test geometry")
plt.savefig("ring_geometry.png", bbox_inches="tight", dpi=150)

print("Points shape: ", pinn.points.interior.coords.shape)
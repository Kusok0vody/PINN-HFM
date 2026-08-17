import sys
import torch
import torch.nn as nn

sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))

from geometry.geom                      import Geometry
from geometry.sampler                   import Sampler
from network.net                        import Net
from network.activations                import ActivationFactory, Sine
from physics.problems.lotka_volterra    import LotkaVolterra
from training.trainer                   import Trainer
from pinn                               import PINN

torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type != "cpu":
    torch.cuda.set_device(device)
print(f"Device: {device}")

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

N_PARAMS_BATCH = 20

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

G0 = 0.8
H0 = 0.4
P0 = 0.2

N_IC  = 1
N_PDE = 256

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

sine = ActivationFactory(Sine, omega=1.0, trainable=True)
net = Net(
    x_dim=1, mu_dim=8,
    dx=128, dmu=128, d_h=64,
    encoder_layers=4, trunk_layers=4, head_layers=3, film_layers=2,
    activation=sine,
    encoder_activation=nn.Tanh,
    trunk_activation=sine,
    head_activation=sine,
    film_activation=nn.Tanh,
    outputs_config={"G": {}, "H": {}, "P": {}},
    use_film=True,
    use_fourier=False,
)

physics = LotkaVolterra(dim=1, has_time=False, device=device)
physics.setParameters(
    params=[PARAMS],
    boundaries=bounds,
    limits=params_definition,
    initial=None,
)

pinn = PINN(
    net, physics, samp,
    n_refine=1,
    adaptive_pde=True,
    adaptive_bc=False,
    adaptive_ic=False,
    device=device,
)

N_ITERS = 30000

trainer = Trainer(
    pinn=pinn,
    lr=1e-3,
    n_iter=N_ITERS,
    resample_every=25,
    checkpoint_every=500,
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

print(f"=== Training ({N_ITERS} iterations) ===")
# trainer.train()
trainer.train_expanding_horizon(
    T_target=30.0,
    n_stages=8,
    n_iter_per_stage=lambda k, N: 1500 + 1000 * (k - 1),
)

trainer.finetune_lbfgs(n_outer=50, max_iter=100, n_cycles=2)

print(trainer.adaptive_weights)
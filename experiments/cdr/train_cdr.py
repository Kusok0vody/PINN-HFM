import sys
import math
import torch
import torch.nn as nn

sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))

from geometry.geom        import Geometry
from geometry.sampler     import Sampler
from network.net          import Net
from network.activations  import ActivationFactory, Sine, Morlet
from physics.problems.cdr import CDR1D
from training.trainer     import Trainer
from pinn                 import PINN

torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

X_MIN, X_MAX = 0.0, 2.0 * math.pi
T_MIN, T_MAX = 0.0, 1.0

N_BC, N_PDE, N_IC = 256, 1024, 2

zero_tx = lambda t, x: torch.zeros_like(x)
bc_periodic = {"u": {"type": "dirichlet", "value": zero_tx}}

bounds = {
    "left": {
        "p": [0.0, 1.0],
        "x": lambda p: torch.zeros_like(p),
        "N": N_BC,
        "periodic": True, "with": "right",
        "bc": bc_periodic,
    },
    "right": {
        "p": [0.0, 1.0],
        "x": lambda p: torch.full_like(p, X_MAX),
        "N": N_BC,
        "periodic": True, "with": "left",
        "bc": bc_periodic,
    },
}

geo  = Geometry(bounds, dim=1, has_time=True, T=[T_MIN, T_MAX])
samp = Sampler(geo, n_interior=N_PDE, n_boundary=N_BC, n_initial=N_IC)

net = Net(
    x_dim=2, mu_dim=3,
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
        "ux": {"activation": nn.Tanh},
    },
    use_film=True,
    use_fourier=False,
)

# params = [{"beta": 2.0, "nu": 0.1, "rho": 4.0}]
params = []
for b in [0.0, 1.0, 2.0, 5.0, 10.0]:
    for r in [0.0, 2.0, 5.0, 10]:
        params.append({"beta": b, "nu": 0.1, "rho": r})

limits = {
    "beta": {"min": 0.0,  "max": 10.0, "N": 20, "scale": "linear"},
    "nu":   {"min": 0.0,  "max": 10.0, "N": 20, "scale": "linear"},
    "rho":  {"min": 0.0,  "max": 10.0, "N": 20, "scale": "linear"},
}

physics = CDR1D(dim=1, has_time=True, device=device)
physics.setParameters(
    params=params,
    boundaries=bounds,
    # initial={
    #     "u": lambda x: torch.ones_like(x) + torch.sin(x),
    # },
    limits=limits,
    hard_ic=True,
)

pinn = PINN(
    net, physics, samp,
    n_refine=10,
    adaptive_pde=True,
    adaptive_bc=False,
    adaptive_ic=False,
    device=device
)
print(pinn)

n_iters = 30000
start   = 0

trainer = Trainer(
    pinn=pinn,
    lr=1e-3,
    n_iter=n_iters,
    resample_every=1000,
    checkpoint_every=500,
    gradnorm_every=200,
    lra_alpha=0.99,
    checkpoint_path="checkpoints",
    run_name="cdr",
    save_final=False,
    logger="tqdm",
    device=device,
    start_step=start,
    param_every=0,
)

trainer.train()

# trainer.train_expanding_horizon(
#     T_target=1.0,
#     n_stages=5,
#     n_iter_per_stage=4000,
#     # n_iter_per_stage=lambda k, N: 2000 + 2000 * k,
# )

trainer.finetune_lbfgs(n_outer=50, max_iter=100, n_cycles=2)

print(trainer.adaptive_weights)
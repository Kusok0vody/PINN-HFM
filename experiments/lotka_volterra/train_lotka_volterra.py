"""
Three-species Lotka-Volterra, swept over every parameter it has.

Eleven axes: the eight rates and the three initial values. Sweeping the Cauchy
data is what makes the hard initial condition possible rather than a
restriction — the ansatz reads u0 out of the parameter vector, so it satisfies
a different initial condition for every setting, and no particular one is baked
into the weights.

The equation is enforced on collocation points, the initial condition holds
identically by construction, and a data term anchors the trajectory against the
integrated reference. That reference is not a closed form but an RK4
integration accurate to about 1e-11 over these ranges, which is eight orders
below anything the network will be judged against.
"""
import sys
import pathlib

import torch
import torch.nn as nn

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "src"))
sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from geometry.geom                      import Geometry
from geometry.sampler                   import Sampler
from network.net                        import Net
from network.activations                import ActivationFactory, Sine
from physics.problems.lotka_volterra    import LotkaVolterra
from training.trainer                   import Trainer
from pinn                               import PINN

import sweep

torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type != "cpu":
    torch.cuda.set_device(device)
print(f"Device: {device}")

T_END = sweep.T_END

N_PARAMS_BATCH = 32
N_PDE  = 256
N_DATA = 256
N_ITERS = 30000

# Shared with the evaluation script, so the two cannot drift apart about which
# ranges this run was trained on.
params_definition = sweep.limits(N_PARAMS_BATCH)
bounds = sweep.bounds()

geo  = Geometry(bounds, dim=1, has_time=False)
samp = Sampler(geo, n_interior=N_PDE, n_boundary=0)

pts = samp.sample()
print(f"interior: {pts.interior.coords.shape}")

sine = ActivationFactory(Sine, omega=1.0, trainable=True)
net = Net(
    x_dim=1, mu_dim=len(params_definition),
    dx=128, dmu=128, d_h=64,
    encoder_layers=4, trunk_layers=4, head_layers=3, film_layers=1,
    activation=sine,
    encoder_activation=nn.Tanh,
    trunk_activation=sine,
    head_activation=sine,
    film_activation=nn.Tanh,
    outputs_config={"G": {}, "H": {}, "P": {}},
    use_film=True,
    use_fourier=False,
)

physics = LotkaVolterra(dim=1, has_time=False, hard_ic=True, device=device)

# The first batch comes from the sweep rather than from a hand-written point,
# so it gets the same treatment every later batch does — including pinned ends
# on each of the eleven axes.
physics.setParameters(
    params=[sweep.midpoint()],
    boundaries=bounds,
    limits=params_definition,
    initial=None,
)
physics._resample_parameters()

pinn = PINN(
    net, physics, samp,
    n_refine=1,
    adaptive_pde=True,
    adaptive_bc=False,
    adaptive_ic=False,
    scale_free_pde=True,
    device=device,
)

pinn.set_data(None, n_points=N_DATA, seed=42)

trainer = Trainer(
    pinn=pinn,
    lr=1e-3,
    n_iter=N_ITERS,
    resample_every=25,
    param_every=100,
    checkpoint_every=200,
    gradnorm_every=200,
    lra_alpha=0.01,
    use_data=True,
    checkpoint_path="checkpoints",
    run_name="lotka_volterra",
    save_final=True,
    logger="tqdm",
    device=device,
    start_step=0,
)

print(f"=== Training ({N_ITERS} iterations) ===")
trainer.train()
print(trainer.adaptive_weights)

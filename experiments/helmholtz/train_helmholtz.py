import sys
import math
import torch
import torch.nn as nn

sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))

from geometry.geom                import Geometry
from geometry.sampler             import Sampler
from network.net                  import Net
from network.activations          import ActivationFactory, Sine
from physics.problems.helmholtz import helmholtz2D_annulus
from training.trainer             import Trainer
from pinn                         import PINN
from validation.references        import helmholtz_annulus_resonances

torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type != "cpu":
    torch.cuda.set_device(device)
print(f"Device: {device}")

R = 3.0
r = 1.0

N_r    = 128
N_R    = 64
N_PDE  = 4096
N_DATA = 256

bounds = {}

for i in range(8):
    p_start = i / 8.0
    p_end   = (i + 1) / 8.0
    bounds[f"rq{i+1}"] = {
        "p": [p_start, p_end],
        "x": lambda p, r=r: r * torch.cos(2 * math.pi * p),
        "y": lambda p, r=r: r * torch.sin(2 * math.pi * p),
        "N": N_r,
        "bc": {
            "u": {"type": "dirichlet", "value": lambda x, y: torch.zeros_like(x)},
        },
    }

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
samp = Sampler(geo, n_interior=N_PDE, n_boundary=N_R)

pts = samp.sample()
print(f"interior: {pts.interior.coords.shape}")
for name, b in pts.boundaries.items():
    print(f"boundary '{name}': {b.coords.shape}")

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
    outputs_config={"u": {"activation": ActivationFactory(Sine, omega=1.0, trainable=True)}},
    use_film=True,
    film_layers=1,
    use_fourier=False,
)

K_MIN = 1.0
K_MAX = 20.0
N_K   = 32

ks = torch.linspace(K_MIN, K_MAX, N_K)
parameters = [{"k": k.item()} for k in ks]
limits = {"k": {"min": K_MIN, "max": K_MAX, "N": N_K, "scale": "linear"}}

# Dirichlet eigenvalues inside the sweep are no longer an error: the physics
# rejects settings whose solution is amplified past AMPLIFICATION_MAX and the
# sweep redraws them, so the poles are holes in the range rather than a reason
# to stop. Worth saying out loud, though — the family is genuinely
# discontinuous across each one, the exact solution changes sign there, and no
# parameterisation smooth in k spans that. Points evaluated inside a hole are
# wrong for that reason, not for one training could fix.
poles = helmholtz_annulus_resonances(K_MIN, K_MAX, r, R, 8)
if poles:
    print(f"sweep k in [{K_MIN}, {K_MAX}] with {N_K} settings; Dirichlet "
          f"eigenvalues at {[round(v, 3) for v in poles]} are excluded by the "
          f"physics and redrawn")
else:
    print(f"sweep k in [{K_MIN}, {K_MAX}], {N_K} settings, no eigenvalues inside")

physics = helmholtz2D_annulus(dim=2, has_time=False, hard_bc=True, device=device)
physics.setParameters(
    params=parameters,
    boundaries=bounds,
    initial=None,
    limits=limits
)

# from network.net import Net as _N
# _N.load_weights(net, torch.load("checkpoints/warm.pt", weights_only=False)["net"])
# The batch above came from a linspace, which knows nothing about which
# settings are worth training on: over [1, 20] it lands on k = 6.516, whose
# exact solution reaches 988 against a boundary datum of 1. Drawing the first
# batch through the sweep instead applies the same rejection every later batch
# gets, and pins the ends of the range while it is at it.
# physics._resample_parameters()
# print("initial settings:",
#       [round(v, 3) for v in physics.par.tensor.flatten().tolist()])

pinn = PINN(
    net, physics, samp,
    n_refine=10,
    adaptive_pde=True,
    adaptive_bc=False,
    adaptive_ic=False,
    scale_free_pde=True,
    device=device,
)

pinn.set_data(None, n_points=N_DATA, seed=42)

N_ITERS = 20000

trainer = Trainer(
    pinn=pinn,
    lr=1e-3,
    n_iter=N_ITERS,
    resample_every=1000,
    param_every=1000,
    checkpoint_every=100,
    gradnorm_every=200,
    lra_alpha=0.01,
    use_data=True,
    checkpoint_path="checkpoints",
    run_name="helmholtz",
    save_final=True,
    logger="tqdm",
    device=device,
    start_step=0,
)

print(f"=== Training ({N_ITERS} iterations) ===")
trainer.train()
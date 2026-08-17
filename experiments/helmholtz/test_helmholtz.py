"""
Picture of a trained Helmholtz run: network, exact solution, difference.

Three panels rather than one, because a single field is unreadable on its own —
the solution of this problem has eight lobes whose sign is opposite to the
boundary datum above them and whose amplitude exceeds it, and none of that looks
right until the exact solution is next to it.

The figure comes from eval_helmholtz.plot_comparison, so the picture and the
numbers can never drift apart.
"""
import sys
import pathlib

import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from network.net import Net
from geometry.geom import Geometry
from geometry.sampler import Sampler
from physics.problems.helmholtz import helmholtz2D_annulus
from pinn import PINN

from eval_helmholtz import build_bounds, plot_comparison, resolve_checkpoint

CHECKPOINT = "checkpoints/helmholtz"     # a ckpt_<step>.pt or the directory
KS         = [10.0]                      # must be what the run was trained on
N          = 601                         # cartesian resolution of the figure
OUTDIR     = "figures"

torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

path = resolve_checkpoint(CHECKPOINT)
net  = Net.from_checkpoint(str(path), device=device)
net.eval()
step = torch.load(path, map_location="cpu", weights_only=False)["step"]
print(f"checkpoint {path}, step {step}")

bounds  = build_bounds()
samp    = Sampler(Geometry(bounds, dim=2, has_time=False), n_interior=16, n_boundary=8)
physics = helmholtz2D_annulus(dim=2, has_time=False, device=device)
physics.setParameters(params=[{"k": k} for k in KS], boundaries=bounds)

# The checkpoint carries the input rescaling it was trained under; recomputing
# it here would silently feed the weights coordinates they never saw. Predicting
# through PINN rather than calling net() applies the output transforms and the
# hard-constraint ansatz, which the raw network output does not include.
pinn = PINN(net, physics, samp, autoscale_inputs=False, device=device)

for k in KS:
    print(plot_comparison(pinn, k, N, OUTDIR, device))

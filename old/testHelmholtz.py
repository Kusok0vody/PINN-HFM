import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("src/")

from training.trainer import Trainer
from physics.problems.Helmholtz  import helmholtz2D_annulus

# -----------------------------
# device
# -----------------------------
torch.manual_seed(42)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

# -----------------------------
# load model
# -----------------------------
CHECKPOINT = "checkpoints/helmholtz/ckpt_20000.pt"
net, step = Trainer.load_checkpoint(path=CHECKPOINT, device=device)
net.eval()

# -----------------------------
# grid in polar coordinates
# -----------------------------
N = 1000
r = 1.0
R = 3.0

xs = torch.linspace(-R, R, N)
ys = torch.linspace(-R, R, N)
YY, XX = torch.meshgrid(ys, xs, indexing="ij")

coords = torch.cat([YY.reshape(-1, 1), XX.reshape(-1, 1)], dim=1).to(device)

# Physical parameters
parameters = [{"k": 5}]
print(parameters)

physics = helmholtz2D_annulus(dim=2, has_time=False, device=device)
physics.setParameters(
    params=parameters,
    boundaries={},
    initial=None
)

# -----------------------------
# forward pass
# -----------------------------
with torch.no_grad():
    raw = net(coords, physics.par.tensor) 
    u = raw["u"].squeeze(1).cpu().numpy()

u = u.reshape(N, N)

# -----------------------------
# mask outside annulus (optional safety)
# -----------------------------
XY = (XX**2 + YY**2)
# mask = (XY.numpy() >= r**2) & ((XX**2 + YY**2).numpy() <= R**2)
mask = ((XX**2 + YY**2).numpy() <= R**2)
u_masked = np.where(mask, u, np.nan)

# -----------------------------
# plot
# -----------------------------
plt.figure(figsize=(7, 6), dpi=150)

im = plt.imshow(
    u_masked,
    origin="lower",
    cmap="rainbow",
    extent=[-R, R, -R, R],
    interpolation="none"
)

plt.colorbar(im, label="u(x,y)")

plt.title(f"PINN solution (Laplace ring), step {step}")
plt.xlabel("x")
plt.ylabel("y")

plt.gca().set_aspect("equal")

# ring outline (optional)
# circle1 = plt.Circle((0, 0), r, color="black", fill=False, linewidth=1)
circle2 = plt.Circle((0, 0), R, color="black", fill=False, linewidth=1)
# plt.gca().add_patch(circle1)
plt.gca().add_patch(circle2)

plt.savefig("pinn_laplace_ring.png", bbox_inches="tight", dpi=300)
plt.show()

print("Saved → pinn_laplace_ring.png")
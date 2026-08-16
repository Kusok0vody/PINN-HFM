import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))

from training.trainer import Trainer
from physics.problems.helmholtz  import helmholtz2D_annulus

torch.manual_seed(42)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

CHECKPOINT = "checkpoints/helmholtz/ckpt_20000.pt"
net, step = Trainer.load_checkpoint(path=CHECKPOINT, device=device)
net.eval()

N = 1000
r = 1.0
R = 3.0

xs = torch.linspace(-R, R, N)
ys = torch.linspace(-R, R, N)
YY, XX = torch.meshgrid(ys, xs, indexing="ij")

coords = torch.cat([YY.reshape(-1, 1), XX.reshape(-1, 1)], dim=1).to(device)

parameters = [{"k": 5}]
print(parameters)

physics = helmholtz2D_annulus(dim=2, has_time=False, device=device)
physics.setParameters(
    params=parameters,
    boundaries={},
)

with torch.no_grad():
    raw = net(coords, physics.par.tensor) 
    u = raw["u"].squeeze(1).cpu().numpy()

u = u.reshape(N, N)

XY = (XX**2 + YY**2)
mask = ((XX**2 + YY**2).numpy() <= R**2)
u_masked = np.where(mask, u, np.nan)

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

circle2 = plt.Circle((0, 0), R, color="black", fill=False, linewidth=1)
plt.gca().add_patch(circle2)

plt.savefig("helmholtz_annulus.png", bbox_inches="tight", dpi=300)
plt.show()

print("Saved --> helmholtz_annulus.png")
import sys
import math
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))

import os
os.makedirs("images", exist_ok=True)

from geometry.geom    import Geometry
from geometry.sampler import Sampler

N_POINTS_STAR = 5
N_BOUND       = 1500
N_INIT        = 1000
N_PDE         = 2500

torch.manual_seed(0)

def star_r(p):
    angle = 2 * math.pi * p
    osc   = torch.cos(N_POINTS_STAR * angle)
    return 0.4 + 0.5 * (osc + 1) / 2

star_bounds = {
    "outer": {
        "p": [0.0, 1.0],
        "x": lambda p: star_r(p) * torch.cos(2 * math.pi * p),
        "y": lambda p: star_r(p) * torch.sin(2 * math.pi * p),
        "N": N_BOUND,
    },
}

geo  = Geometry(star_bounds, dim=2, has_time=True, T = [0, 1])
samp = Sampler(geo, n_interior=N_PDE, n_boundary=N_BOUND, n_initial=N_INIT)
pts  = samp.sample()

p_contour = torch.linspace(0.0, 1.0, 2000)
cx = star_r(p_contour) * torch.cos(2 * math.pi * p_contour)
cy = star_r(p_contour) * torch.sin(2 * math.pi * p_contour)

mpl.rcParams.update({
    "font.size":      13,
    "axes.labelsize": 14,
    "font.family":    "DejaVu Sans",
})

fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=200)
ax.set_aspect("equal")
ax.grid(True, linewidth=0.35, alpha=0.4, color="gray")

ic = pts.interior.coords
ax.scatter(
    ic[:, -1].numpy(), ic[:, -2].numpy(),
    s=9, color="#e05252", alpha=0.45, linewidths=0,
    label=None,
    zorder=2, rasterized=True,
)

initc = pts.initial.coords
ax.scatter(
    initc[:, -1].numpy(), initc[:, -2].numpy(),
    s=9, color="#528fe0", alpha=0.45, linewidths=0,
    label=None,
    zorder=2, rasterized=True,
)

b  = pts.boundaries["outer"]
xb = b.coords[:, -1].numpy()
yb = b.coords[:, -2].numpy()
ax.scatter(
    xb, yb,
    s=18, color="#22aa55", edgecolors="black", linewidths=0.35,
    zorder=5,
)

stride = 25
nx = b.nx[::stride, 0].numpy()
ny = b.ny[::stride, 0].numpy()
ax.quiver(
    xb[::stride], yb[::stride], nx, ny,
    color="#555555", scale=10, scale_units="xy",
    width=0.004, alpha=0.9,
    zorder=4,
)

ax.plot(cx.numpy(), cy.numpy(), "k-", linewidth=1.0, zorder=6)

lim = 1.05
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")

plt.tight_layout()
plt.savefig("images/star_geometry.png", bbox_inches="tight", dpi=300)
print("Saved --> images/star_geometry.png")

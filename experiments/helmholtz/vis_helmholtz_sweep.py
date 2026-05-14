import sys
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))

import os
os.makedirs("images", exist_ok=True)

from physics.problems.helmholtz import helmholtz2D_annulus
from training.trainer import Trainer

CHECKPOINT = "checkpoints/helmholtz/ckpt_20000.pt"
OUTPUT_PNG = "images/helmholtz_sweep.png"
KS         = [1, 2, 5, 10]
N          = 500
R          = 3.0
r          = 1.0
CMAP       = "RdBu_r"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

xs = torch.linspace(-R, R, N)
ys = torch.linspace(-R, R, N)
YY, XX = torch.meshgrid(ys, xs, indexing="ij")
coords = torch.cat([YY.reshape(-1, 1), XX.reshape(-1, 1)], dim=1).to(device)

rr = (XX**2 + YY**2).numpy()
mask_annulus = (rr >= r**2 - 1e-4) & (rr <= R**2 + 1e-4)
x_lin = np.linspace(-R, R, N)
y_lin = np.linspace(-R, R, N)

net, step = Trainer.load_checkpoint(path=CHECKPOINT, device=device)
net.eval()
print(f"Step: {step}")

results = {}
for k in KS:
    physics = helmholtz2D_annulus(dim=2, has_time=False, device=device)
    physics.setParameters(params=[{"k": k}], boundaries={}, initial=None)
    with torch.no_grad():
        raw = net(coords, physics.par.tensor)
    u = raw["u"].squeeze(1).cpu().numpy().reshape(N, N)
    results[k] = np.where(mask_annulus, u, np.nan)
    print(f"  k={k}  umin={np.nanmin(results[k]):.3f}  umax={np.nanmax(results[k]):.3f}")

all_vals = np.concatenate([v[mask_annulus] for v in results.values()])
vabs = np.nanpercentile(np.abs(all_vals), 99)
vmin, vmax = -vabs, vabs

theta = np.linspace(0, 2 * np.pi, 600)

fig = plt.figure(figsize=(10.5, 10.0), dpi=200)
gs = gridspec.GridSpec(
    2, 2,
    figure=fig,
    wspace=0.12,
    hspace=0.18,
    left=0.06, right=0.91, top=0.93, bottom=0.05,
)

axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]

for ax, k in zip(axes, KS):
    u_masked = results[k]

    im = ax.imshow(
        u_masked,
        origin="lower",
        cmap=CMAP,
        extent=[-R, R, -R, R],
        interpolation="bilinear",
        vmin=vmin,
        vmax=vmax,
    )

    levels = np.linspace(vmin * 0.8, vmax * 0.8, 14)
    ax.contour(x_lin, y_lin, u_masked, levels=levels,
               colors="white", linewidths=0.45, alpha=0.5)

    ax.plot(R * np.cos(theta), R * np.sin(theta), "k-", linewidth=1.2)
    ax.plot(r * np.cos(theta), r * np.sin(theta), "k-", linewidth=1.2)

    for i in range(8):
        t0 = i / 8.0 * 2 * np.pi
        t1 = (i + 1) / 8.0 * 2 * np.pi
        th = np.linspace(t0, t1, 60)
        col = "#cc1111" if (i % 2 == 0) else "#1133cc"
        r_in, r_out = R, R + 0.22
        xs_arc = np.concatenate([r_in * np.cos(th), r_out * np.cos(th[::-1])])
        ys_arc = np.concatenate([r_in * np.sin(th), r_out * np.sin(th[::-1])])
        ax.fill(xs_arc, ys_arc, color=col, alpha=0.85, zorder=5)

    ax.set_aspect("equal")
    ax.set_xlim(-R * 1.22, R * 1.22)
    ax.set_ylim(-R * 1.22, R * 1.22)
    ax.set_title(rf"$k = {k}$", fontsize=14, pad=5)
    ax.tick_params(labelsize=9)

    row, col_idx = divmod(KS.index(k), 2)
    if col_idx == 0:
        ax.set_ylabel(r"$y$", fontsize=12)
    else:
        ax.set_yticklabels([])
    if row == 1:
        ax.set_xlabel(r"$x$", fontsize=12)
    else:
        ax.set_xticklabels([])

cbar_ax = fig.add_axes([0.93, 0.05, 0.025, 0.88])
cb = fig.colorbar(im, cax=cbar_ax)
cb.set_label(r"$u(x,\,y)$", fontsize=13)
cb.ax.tick_params(labelsize=10)

fig.suptitle(
    r"$\Delta u + k^2 u = 0$ in ring - PINN for various $k$",
    fontsize=14,
    y=0.97,
)

plt.savefig(OUTPUT_PNG, bbox_inches="tight", dpi=300)
print(f"Saved --> {OUTPUT_PNG}")

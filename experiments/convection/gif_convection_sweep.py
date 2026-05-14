import sys
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))

import os
os.makedirs("gifs", exist_ok=True)

from physics.problems.convection1D import convection1D
from training.trainer import Trainer

CHECKPOINT   = "checkpoints/convection/ckpt_20000.pt"
OUTPUT_GIF   = "gifs/convection_beta_sweep.gif"
FPS          = 4

N_GRID = 180
N_TIME = 180

BETAS_SWEEP = (
    [1.0] * 3
    + list(np.arange(1.0, 20.5, 0.5))
    + [20.0] * 3
)

CMAP_SOLUTION = "rainbow"
CMAP_ERROR    = "hot_r"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type != "cpu":
    torch.cuda.set_device(device)
print(f"Device: {device}")

xs = torch.linspace(0.0, 2 * math.pi, N_GRID)
ts = torch.linspace(0.0, 1.0, N_TIME)
TT, XX = torch.meshgrid(ts, xs, indexing="ij")
coords = torch.cat([TT.reshape(-1, 1), XX.reshape(-1, 1)], dim=1).to(device)
TT_np = TT.numpy()
XX_np = XX.numpy()

print(f"Загрузка {CHECKPOINT}...")
net, step = Trainer.load_checkpoint(path=CHECKPOINT, device=device)
net.eval()
print(f"Шаг: {step}")

u_min_global =  1.0 - 1.0
u_max_global =  1.0 + 1.0
err_max_global = 0.5

def compute_frame(beta_val):
    ph = convection1D(dim=1, has_time=True, device=device)
    ph.setParameters(
        params=[{"beta": beta_val}],
        boundaries={},
        initial={"u": lambda x: torch.ones_like(x) + torch.sin(x)},
    )
    with torch.no_grad():
        raw  = net(coords, ph.par.tensor)
        pred = ph.apply_transforms(raw)
    u_pred  = pred["u"].squeeze(1).cpu().reshape(N_TIME, N_GRID).numpy()
    u_exact = 1.0 + np.sin(XX_np - beta_val * TT_np)
    u_err   = np.abs(u_pred - u_exact)
    l2_rel  = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)
    return u_pred, u_exact, u_err, l2_rel

print(f"Compute {len(set(BETAS_SWEEP))} unique betas...")
cache = {}
for b in sorted(set(BETAS_SWEEP)):
    cache[b] = compute_frame(b)
    print(f"  beta={b:.1f}  L2={cache[b][3]:.4f}", end="\r")
print(f"\Done.")

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

fig = plt.figure(figsize=(11, 4.2), dpi=130)
fig.patch.set_facecolor("#0d0d0d")

gs = gridspec.GridSpec(
    1, 4,
    figure=fig,
    width_ratios=[1, 1, 1, 0.055],
    wspace=0.12,
    left=0.06, right=0.91, top=0.82, bottom=0.13,
)

ax_pred = fig.add_subplot(gs[0, 0])
ax_exact = fig.add_subplot(gs[0, 1])
ax_err  = fig.add_subplot(gs[0, 2])
cbar_ax = fig.add_subplot(gs[0, 3])

extent = [0, 1, 0, 2 * math.pi]

ax_style = dict(facecolor="#0d0d0d")

blank = np.zeros((N_GRID, N_TIME))

im_pred = ax_pred.imshow(
    blank, origin="lower", extent=extent, aspect="auto",
    cmap=CMAP_SOLUTION, vmin=u_min_global, vmax=u_max_global,
    interpolation="bilinear",
)
im_exact = ax_exact.imshow(
    blank, origin="lower", extent=extent, aspect="auto",
    cmap=CMAP_SOLUTION, vmin=u_min_global, vmax=u_max_global,
    interpolation="bilinear",
)
im_err = ax_err.imshow(
    blank, origin="lower", extent=extent, aspect="auto",
    cmap=CMAP_ERROR, vmin=0, vmax=err_max_global,
    interpolation="bilinear",
)

for ax, title in zip(
    [ax_pred, ax_exact, ax_err],
    ["PINN prediction", "Exact solution", "Absolute error"]
):
    ax.set_facecolor("#0d0d0d")
    ax.set_title(title, color="white", fontsize=10, pad=5)
    ax.set_xlabel("$t$", color="#aaaaaa", fontsize=10)
    ax.tick_params(colors="#777777", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

ax_pred.set_ylabel("$x$", color="#aaaaaa", fontsize=10)
ax_pred.set_yticks([0, math.pi, 2 * math.pi])
ax_pred.set_yticklabels(["$0$", "$\pi$", "$2\pi$"], color="#aaaaaa", fontsize=8)

for ax in [ax_exact, ax_err]:
    ax.set_yticks([])

for ax in [ax_pred, ax_exact, ax_err]:
    ax.set_xticks([0, 0.5, 1.0])

cb = fig.colorbar(im_pred, cax=cbar_ax, label="$u$")
cb.ax.yaxis.label.set_color("white")
cb.ax.tick_params(colors="#777777", labelsize=8)
cbar_ax.set_facecolor("#0d0d0d")

from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax_err)
err_cbar_ax = divider.append_axes("right", size="6%", pad=0.06)
err_cbar_ax.set_facecolor("#0d0d0d")
cb_err = fig.colorbar(im_err, cax=err_cbar_ax)
cb_err.ax.tick_params(colors="#777777", labelsize=7)

suptitle = fig.suptitle("", color="white", fontsize=13, fontweight="bold", y=0.97)

l2_text = fig.text(
    0.5, 0.01, "",
    ha="center", va="bottom",
    color="#aaaaaa", fontsize=9,
    transform=fig.transFigure,
)

def update(frame_idx):
    beta = BETAS_SWEEP[frame_idx]
    u_pred, u_exact, u_err, l2 = cache[beta]

    im_pred.set_data(u_pred.T)
    im_exact.set_data(u_exact.T)
    im_err.set_data(u_err.T)

    suptitle.set_text(rf"1D convection  $\beta = {beta:.1f}$  |  step {step}")
    l2_text.set_text(rf"$L_2^{{rel}}$ = {l2:.4f}")

    return [im_pred, im_exact, im_err, suptitle, l2_text]

ani = FuncAnimation(
    fig, update,
    frames=len(BETAS_SWEEP),
    interval=1000 // FPS,
    blit=False,
)

writer = PillowWriter(fps=FPS)
ani.save(OUTPUT_GIF, writer=writer)
plt.close(fig)
print(f"Saved --> {OUTPUT_GIF}")
"""
Картинка: решение уравнения Гельмгольца в кольце (2D).

Показывает:
  - цветовую карту PINN решения u(x,y) в кольцевой области
  - изолинии u = const (белые)
  - сектора внешней границы, окрашенные по значению BC (+1 / -1)
  - аннотацию уравнения и радиусов

Требует чекпоинт: checkpoints/helmholtz/ckpt_20000.pt
"""

import sys
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.append("src/")

import os
os.makedirs("images", exist_ok=True)

from physics.problems.Helmholtz import helmholtz2D_annulus
from training.trainer import Trainer

# ── Настройки ─────────────────────────────────────────────────────────────────
CHECKPOINT = "checkpoints/helmholtz/ckpt_20000.pt"
OUTPUT_PNG = "images/helmholtz_2d.png"
K_PLOT     = 1.0
N          = 700
R          = 3.0
r          = 1.0
CMAP       = "RdBu_r"
N_CONTOURS = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Сетка ─────────────────────────────────────────────────────────────────────
xs = torch.linspace(-R, R, N)
ys = torch.linspace(-R, R, N)
YY, XX = torch.meshgrid(ys, xs, indexing="ij")
coords = torch.cat([YY.reshape(-1, 1), XX.reshape(-1, 1)], dim=1).to(device)

rr = (XX**2 + YY**2).numpy()
mask_annulus = (rr <= R**2 + 1e-4)

# ── Модель ────────────────────────────────────────────────────────────────────
net, step = Trainer.load_checkpoint(path=CHECKPOINT, device=device)
net.eval()
print(f"Шаг: {step}")

physics = helmholtz2D_annulus(dim=2, has_time=False, device=device)
physics.setParameters(params=[{"k": K_PLOT}], boundaries={}, initial=None)

with torch.no_grad():
    raw = net(coords, physics.par.tensor)
u = raw["u"].squeeze(1).cpu().numpy().reshape(N, N)
u_masked = np.where(mask_annulus, u, np.nan)

# ── График ────────────────────────────────────────────────────────────────────
vabs = np.nanpercentile(np.abs(u_masked), 99)
vmin, vmax = -vabs, vabs

fig, ax = plt.subplots(figsize=(6.8, 6.5), dpi=200)

im = ax.imshow(
    u_masked,
    origin="lower",
    cmap=CMAP,
    extent=[-R, R, -R, R],
    interpolation="bilinear",
    vmin=vmin,
    vmax=vmax,
)

# Изолинии
x_lin = np.linspace(-R, R, N)
y_lin = np.linspace(-R, R, N)
levels = np.linspace(vmin * 0.85, vmax * 0.85, N_CONTOURS)
cs = ax.contour(
    x_lin, y_lin, u_masked,
    levels=levels,
    colors="white",
    linewidths=0.55,
    alpha=0.55,
)

# Границы
theta = np.linspace(0, 2 * np.pi, 700)
ax.plot(R * np.cos(theta), R * np.sin(theta), "k-", linewidth=1.6, zorder=10)
# ax.plot(r * np.cos(theta), r * np.sin(theta), "k-", linewidth=1.6, zorder=10)

# Секторы BC — цветные полосы за внешней границей
for i in range(8):
    t0  = i / 8.0 * 2 * np.pi
    t1  = (i + 1) / 8.0 * 2 * np.pi
    th  = np.linspace(t0, t1, 80)
    val = 1.0 if (i % 2 == 0) else -1.0
    col = "#cc1111" if val > 0 else "#1133cc"
    # Заполненная дуга
    r_in, r_out = R, R + 0.28
    xs_arc = np.concatenate([r_in * np.cos(th), r_out * np.cos(th[::-1])])
    ys_arc = np.concatenate([r_in * np.sin(th), r_out * np.sin(th[::-1])])
    ax.fill(xs_arc, ys_arc, color=col, alpha=0.85, zorder=9)

# Метки BC
for i in range(8):
    mid_angle = (i + 0.5) / 8.0 * 2 * np.pi
    xm = (R + 0.48) * math.cos(mid_angle)
    ym = (R + 0.48) * math.sin(mid_angle)
    val = "+1" if (i % 2 == 0) else "−1"
    col = "#aa0000" if (i % 2 == 0) else "#0022aa"
    ax.text(xm, ym, val, ha="center", va="center",
            fontsize=8, fontweight="bold", color=col, zorder=11)

# Аннотация радиусов
# ax.annotate("", xy=(r * math.cos(math.pi * 0.7), r * math.sin(math.pi * 0.7)),
#             xytext=(0, 0),
#             arrowprops=dict(arrowstyle="->", color="black", lw=1.0))
# ax.text(0.35, 0.55, f"$r={r:.0f}$", fontsize=10, color="black",
#         transform=ax.transData)

# ax.annotate("", xy=(R * math.cos(math.pi * 1.4), R * math.sin(math.pi * 1.4)),
#             xytext=(0, 0),
#             arrowprops=dict(arrowstyle="->", color="black", lw=1.0))
# ax.text(-2.6, -1.35, f"$R={R:.0f}$", fontsize=10, color="black")

cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.08)
cb.set_label(r"$u(x,\,y)$", fontsize=13)
cb.ax.tick_params(labelsize=10)

ax.set_aspect("equal")
ax.set_xlim(-R * 1.38, R * 1.38)
ax.set_ylim(-R * 1.38, R * 1.38)
ax.set_xlabel(r"$x$", fontsize=13)
ax.set_ylabel(r"$y$", fontsize=13)
# ax.set_title(
#     rf"$\Delta u + k^2 u = 0$ в кольце, $k = {K_PLOT:.0f}$",
#     fontsize=14, pad=12,
# )

plt.tight_layout()
plt.savefig(OUTPUT_PNG, bbox_inches="tight", dpi=300)
print(f"Сохранено → {OUTPUT_PNG}")

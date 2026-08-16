"""
GIF эволюции решения уравнения Лапласа в кольце в ходе обучения
+ финальная картинка решения.

Геометрия (4 квадранта на каждой границе):
  Внутренняя (r=1): rq1/rq3 → u=+1,  rq2/rq4 → u=0
  Внешняя    (R=3): Rq1/Rq3 → u= 0,  Rq2/Rq4 → u=-1

Требует чекпоинты: checkpoints/laplace/ckpt_*.pt
"""

import sys
import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.append("src/")

os.makedirs("images", exist_ok=True)
os.makedirs("gifs",   exist_ok=True)

from physics.problems.Helmholtz import helmholtz2D_annulus
from training.trainer import Trainer

# ── Настройки ─────────────────────────────────────────────────────────────────
CHECKPOINT_DIR  = "checkpoints/laplace"
CHECKPOINT_STEP = 100
MAX_STEP        = 20000
OUTPUT_GIF      = "gifs/laplace_evolution.gif"
OUTPUT_PNG      = "images/laplace_final.png"
FPS             = 15

N    = 500
R    = 3.0
r    = 1.0
CMAP = "RdBu_r"

# BC на 4 квадрантах (индекс 0..3 соответствует p=[0,.25],[.25,.5],[.5,.75],[.75,1])
INNER_BC = [+1.0,  0.0, +1.0,  0.0]   # rq1..rq4
OUTER_BC = [ 0.0, -1.0,  0.0, -1.0]   # Rq1..Rq4
BC_COLOR = {+1.0: "#cc2222", 0.0: "#aaaaaa", -1.0: "#2255cc"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Декартова сетка ────────────────────────────────────────────────────────────
xs = torch.linspace(-R, R, N)
ys = torch.linspace(-R, R, N)
YY, XX = torch.meshgrid(ys, xs, indexing="ij")
coords = torch.cat([YY.reshape(-1, 1), XX.reshape(-1, 1)], dim=1).to(device)

rr           = (XX**2 + YY**2).numpy()
mask_annulus = (rr >= r**2) & (rr <= R**2)

x_lin = np.linspace(-R, R, N)
y_lin = np.linspace(-R, R, N)
theta = np.linspace(0, 2 * np.pi, 600)

# ── Физика (k=0 → Лаплас) ────────────────────────────────────────────────────
physics = helmholtz2D_annulus(dim=2, has_time=False, device=device)
physics.setParameters(params=[{"k": 0}], boundaries={}, initial=None)

# ── Предсказание ──────────────────────────────────────────────────────────────
def predict(net):
    with torch.no_grad():
        raw = net(coords, physics.par.tensor)
    u = raw["u"].squeeze(1).cpu().numpy().reshape(N, N)
    return np.where(mask_annulus, u, np.nan)

# ── Чекпоинты ─────────────────────────────────────────────────────────────────
steps = [s for s in range(0, MAX_STEP + 1, CHECKPOINT_STEP)
         if os.path.exists(f"{CHECKPOINT_DIR}/ckpt_{s}.pt")]
if not steps:
    raise FileNotFoundError(f"Net chekpointov v {CHECKPOINT_DIR}/ckpt_*.pt")
print(f"Naydeno chekpointov: {len(steps)}  ({steps[0]} ... {steps[-1]})")

# ── Предзагрузка ──────────────────────────────────────────────────────────────
print("Vychislenie predskazaniy...")
cache = {}
for s in steps:
    net, _ = Trainer.load_checkpoint(path=f"{CHECKPOINT_DIR}/ckpt_{s}.pt", device=device)
    net.eval()
    cache[s] = predict(net)
    print(f"  ckpt_{s}.pt ok", end="\r")
print(f"\nGotovo. Generiruyu GIF ({len(steps)} kadrov)...")

# Цветовые пределы
all_vals = np.concatenate([v[mask_annulus] for v in cache.values()])
vabs     = np.nanpercentile(np.abs(all_vals), 98)
vmin, vmax = -vabs, vabs


# ── Вспомогательная функция: нарисовать аннотации BC на осях ─────────────────
def draw_bc_arcs(ax, gap_inner=0.14, gap_outer=0.18, lw_gif=4):
    """Цветные дуги снаружи/внутри границ по значениям BC."""
    for qi in range(4):
        t0 = qi / 4.0 * 2 * np.pi
        t1 = (qi + 1) / 4.0 * 2 * np.pi
        th = np.linspace(t0, t1, 80)

        # Внутренняя граница — дуги чуть внутри (r - gap)
        col_in = BC_COLOR[INNER_BC[qi]]
        ax.plot((r - gap_inner) * np.cos(th), (r - gap_inner) * np.sin(th),
                "-", color=col_in, linewidth=lw_gif, solid_capstyle="butt", zorder=8)

        # Внешняя граница — дуги снаружи (R + gap)
        col_out = BC_COLOR[OUTER_BC[qi]]
        ax.plot((R + gap_outer) * np.cos(th), (R + gap_outer) * np.sin(th),
                "-", color=col_out, linewidth=lw_gif, solid_capstyle="butt", zorder=8)

    ax.plot(R * np.cos(theta), R * np.sin(theta), "k-", linewidth=1.2, zorder=9)
    ax.plot(r * np.cos(theta), r * np.sin(theta), "k-", linewidth=1.2, zorder=9)


# ═══════════════════════════════════════════════════════════════════════════════
#  GIF
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6.2, 6), dpi=150)

im = ax.imshow(
    cache[steps[0]],
    origin="lower", cmap=CMAP,
    extent=[-R, R, -R, R],
    interpolation="bilinear",
    vmin=vmin, vmax=vmax,
)

draw_bc_arcs(ax)

cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
cb.set_label(r"$u(x,\,y)$", fontsize=12)

ax.set_aspect("equal")
ax.set_xlim(-R * 1.25, R * 1.25)
ax.set_ylim(-R * 1.25, R * 1.25)
ax.set_xlabel(r"$x$", fontsize=12)
ax.set_ylabel(r"$y$", fontsize=12)
title = ax.set_title("", fontsize=12, fontweight="bold")


def update(frame_idx):
    s = steps[frame_idx]
    im.set_data(cache[s])
    title.set_text(f"Шаг {s:>5d}/{MAX_STEP}")
    return [im, title]


ani = FuncAnimation(fig, update, frames=len(steps), interval=1000 // FPS, blit=False)
ani.save(OUTPUT_GIF, writer=PillowWriter(fps=FPS))
plt.close(fig)
print(f"Sokhraneno: {OUTPUT_GIF}")

# ═══════════════════════════════════════════════════════════════════════════════
#  Финальная картинка
# ═══════════════════════════════════════════════════════════════════════════════
u_final = cache[steps[-1]]

fig2, ax2 = plt.subplots(figsize=(6.5, 6.2), dpi=200)

im2 = ax2.imshow(
    u_final,
    origin="lower", cmap=CMAP,
    extent=[-R, R, -R, R],
    interpolation="bilinear",
    vmin=vmin, vmax=vmax,
)

# Изолинии
levels = np.linspace(vmin * 0.9, vmax * 0.9, 16)
ax2.contour(x_lin, y_lin, u_final,
            levels=levels, colors="white", linewidths=0.6, alpha=0.55)

draw_bc_arcs(ax2, gap_inner=0.16, gap_outer=0.22, lw_gif=5)

# Метки значений BC
for qi in range(4):
    mid_angle = (qi + 0.5) / 4.0 * 2 * np.pi

    # Внутренняя
    val_in = INNER_BC[qi]
    if val_in != 0.0:
        xm = (r - 0.42) * math.cos(mid_angle)
        ym = (r - 0.42) * math.sin(mid_angle)
        label = f"+{val_in:.0f}" if val_in > 0 else f"{val_in:.0f}"
        ax2.text(xm, ym, label, ha="center", va="center",
                 fontsize=8, fontweight="bold", color=BC_COLOR[val_in], zorder=11)

    # Внешняя
    val_out = OUTER_BC[qi]
    if val_out != 0.0:
        xm = (R + 0.50) * math.cos(mid_angle)
        ym = (R + 0.50) * math.sin(mid_angle)
        label = f"+{val_out:.0f}" if val_out > 0 else f"{val_out:.0f}"
        ax2.text(xm, ym, label, ha="center", va="center",
                 fontsize=8, fontweight="bold", color=BC_COLOR[val_out], zorder=11)

cb2 = fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.08)
cb2.set_label(r"$u(x,\,y)$", fontsize=12)

ax2.set_aspect("equal")
ax2.set_xlim(-R * 1.30, R * 1.30)
ax2.set_ylim(-R * 1.30, R * 1.30)
ax2.set_xlabel(r"$x$", fontsize=12)
ax2.set_ylabel(r"$y$", fontsize=12)
ax2.set_title(
    rf"PINN: уравнение Лапласа в кольце, шаг {steps[-1]}",
    fontsize=12, pad=10,
)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, bbox_inches="tight", dpi=300)
plt.close(fig2)
print(f"Sokhraneno: {OUTPUT_PNG}")

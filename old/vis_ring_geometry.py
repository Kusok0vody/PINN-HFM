"""
Рисунок: область и точки коллокации для задачи Лапласа в кольце.

Геометрия (4 квадранта на каждой границе):
  Внутренняя (r=1): rq1/rq3 → u=+1,  rq2/rq4 → u=0
  Внешняя    (R=3): Rq1/Rq3 → u= 0,  Rq2/Rq4 → u=-1

Запускается без чекпоинтов.
"""

import sys
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.append("src/")

import os
os.makedirs("images", exist_ok=True)

from geometry.geom    import Geometry
from geometry.sampler import Sampler

# ── Параметры ─────────────────────────────────────────────────────────────────
R     = 3.0
r     = 1.0
N_r   = 256    # точек на каждую из 4 внутренних дуг
N_R   = 128    # точек на каждую из 4 внешних дуг
N_PDE = 2048

torch.manual_seed(0)

# BC на 4 квадрантах
INNER_BC = {"rq1": +1.0, "rq2": 0.0, "rq3": +1.0, "rq4": 0.0}
OUTER_BC = {"Rq1":  0.0, "Rq2": -1.0, "Rq3":  0.0, "Rq4": -1.0}
BC_COLOR = {+1.0: "#cc2222", 0.0: "#aaaaaa", -1.0: "#2255cc"}

# ── Геометрия ─────────────────────────────────────────────────────────────────
bounds = {
    "rq1": {
        "p": [0.0, 0.25],
        "x": lambda p: r * torch.cos(2 * math.pi * p),
        "y": lambda p: r * torch.sin(2 * math.pi * p),
        "N": N_r,
        "bc": {"u": {"type": "dirichlet", "value": lambda x, y: torch.ones_like(x)}},
    },
    "rq2": {
        "p": [0.25, 0.5],
        "x": lambda p: r * torch.cos(2 * math.pi * p),
        "y": lambda p: r * torch.sin(2 * math.pi * p),
        "N": N_r,
        "bc": {"u": {"type": "dirichlet", "value": lambda x, y: torch.zeros_like(x)}},
    },
    "rq3": {
        "p": [0.5, 0.75],
        "x": lambda p: r * torch.cos(2 * math.pi * p),
        "y": lambda p: r * torch.sin(2 * math.pi * p),
        "N": N_r,
        "bc": {"u": {"type": "dirichlet", "value": lambda x, y: torch.ones_like(x)}},
    },
    "rq4": {
        "p": [0.75, 1.0],
        "x": lambda p: r * torch.cos(2 * math.pi * p),
        "y": lambda p: r * torch.sin(2 * math.pi * p),
        "N": N_r,
        "bc": {"u": {"type": "dirichlet", "value": lambda x, y: torch.zeros_like(x)}},
    },
    "Rq1": {
        "p": [0.0, 0.25],
        "x": lambda p: R * torch.cos(2 * math.pi * p),
        "y": lambda p: R * torch.sin(2 * math.pi * p),
        "N": N_R,
        "bc": {"u": {"type": "dirichlet", "value": lambda x, y: torch.zeros_like(x)}},
    },
    "Rq2": {
        "p": [0.25, 0.5],
        "x": lambda p: R * torch.cos(2 * math.pi * p),
        "y": lambda p: R * torch.sin(2 * math.pi * p),
        "N": N_R,
        "bc": {"u": {"type": "dirichlet", "value": lambda x, y: -torch.ones_like(x)}},
    },
    "Rq3": {
        "p": [0.5, 0.75],
        "x": lambda p: R * torch.cos(2 * math.pi * p),
        "y": lambda p: R * torch.sin(2 * math.pi * p),
        "N": N_R,
        "bc": {"u": {"type": "dirichlet", "value": lambda x, y: torch.zeros_like(x)}},
    },
    "Rq4": {
        "p": [0.75, 1.0],
        "x": lambda p: R * torch.cos(2 * math.pi * p),
        "y": lambda p: R * torch.sin(2 * math.pi * p),
        "N": N_R,
        "bc": {"u": {"type": "dirichlet", "value": lambda x, y: -torch.ones_like(x)}},
    },
}

geo  = Geometry(bounds, dim=2, has_time=False)
samp = Sampler(geo, n_interior=N_PDE, n_boundary=N_R)
pts  = samp.sample()

# ── Стиль ─────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.size":      13,
    "axes.labelsize": 14,
    "font.family":    "DejaVu Sans",
})

fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=200)
ax.set_aspect("equal")
ax.grid(True, linewidth=0.35, alpha=0.4, color="gray")

# ── Внутренняя область ────────────────────────────────────────────────────────
ic = pts.interior.coords
ax.scatter(
    ic[:, 1].numpy(), ic[:, 0].numpy(),
    s=8, color="#52e067", alpha=0.55, linewidths=0,
    label="Внутренняя область",
    zorder=2, rasterized=True,
)

# ── Граничные точки ───────────────────────────────────────────────────────────
# Для легенды — отслеживаем, добавлена ли уже метка для каждого значения BC
legend_added = {+1.0: False, 0.0: False, -1.0: False}
label_text   = {+1.0: r"Граница ($u = +1$)",
                 0.0: r"Граница ($u = 0$)",
                -1.0: r"Граница ($u = -1$)"}

all_bc = {**INNER_BC, **OUTER_BC}
for name, val in all_bc.items():
    b  = pts.boundaries[name]
    xb = b.coords[:, 1].numpy()
    yb = b.coords[:, 0].numpy()

    label = label_text[val] if not legend_added[val] else None
    legend_added[val] = True

    ax.scatter(
        xb, yb,
        s=30, color=BC_COLOR[val], edgecolors="black", linewidths=0.4,
        label=label, zorder=6,
    )

# ── Нормали ───────────────────────────────────────────────────────────────────
stride = 15
normals_labeled = False
for name in all_bc:
    b  = pts.boundaries[name]
    xb = b.coords[::stride, 1].numpy()
    yb = b.coords[::stride, 0].numpy()
    nx = b.nx[::stride, 0].numpy()
    ny = b.ny[::stride, 0].numpy()
    ax.quiver(
        xb, yb, nx, ny,
        color="#777777", scale=12, scale_units="xy",
        width=0.003, alpha=0.85, zorder=4,
        label="Нормаль" if not normals_labeled else "",
    )
    normals_labeled = True

# ── Окружности ────────────────────────────────────────────────────────────────
theta = np.linspace(0, 2 * np.pi, 600)
ax.plot(R * np.cos(theta), R * np.sin(theta), "k-", linewidth=1.2, zorder=7)
ax.plot(r * np.cos(theta), r * np.sin(theta), "k-", linewidth=1.2, zorder=7)

ax.set_xlim(-R * 1.18, R * 1.18)
ax.set_ylim(-R * 1.18, R * 1.18)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
# ax.set_title("Область и точки коллокации (кольцо)", pad=10)

# ax.legend(fontsize=10, loc="upper right", framealpha=0.92,
#           edgecolor="gray", markerscale=1.3)

plt.tight_layout()
plt.savefig("images/ring_geometry.png", bbox_inches="tight", dpi=300)
print("Sokhraneno: images/ring_geometry.png")

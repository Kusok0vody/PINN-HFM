"""
GIF 1: Эволюция обучения — 6 тепловых карт для beta = 1, 5, 10, 12.5, 15, 20
Кадры: каждые 100 шагов от 0 до 20000
"""

import sys
import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.append("src/")

import os
os.makedirs("gifs", exist_ok=True)

from physics.problems.convection1D import convection1D
from training.trainer import Trainer

# ── Настройки ────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = "checkpoints/convection"
CHECKPOINT_STEP = 100          # шаг между чекпоинтами
MAX_STEP        = 20000
OUTPUT_GIF      = "gifs/convection_evolution.gif"
FPS             = 8            # кадров в секунду

N_GRID = 150
N_TIME = 150

BETAS = [1.0, 5.0, 10.0, 12.5, 15.0, 20.0]
CMAP  = "rainbow"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type != "cpu":
    torch.cuda.set_device(device)
print(f"Device: {device}")

# ── Сетка координат ──────────────────────────────────────────────────────────
xs = torch.linspace(0.0, 2 * math.pi, N_GRID)
ts = torch.linspace(0.0, 1.0, N_TIME)
TT, XX = torch.meshgrid(ts, xs, indexing="ij")
coords_np = torch.cat([TT.reshape(-1, 1), XX.reshape(-1, 1)], dim=1)
coords = coords_np.to(device)

# ── Точные решения (не меняются) ─────────────────────────────────────────────
exact = {}
for beta in BETAS:
    exact[beta] = 1.0 + np.sin(XX.numpy() - beta * TT.numpy())  # (N_TIME, N_GRID)

# Общий vmin/vmax по точным решениям
vmin = min(v.min() for v in exact.values())
vmax = max(v.max() for v in exact.values())

# ── Инициализация физики для каждого beta ────────────────────────────────────
def make_physics(beta_val):
    ph = convection1D(dim=1, has_time=True, device=device)
    ph.setParameters(
        params=[{"beta": beta_val}],
        boundaries={},
        initial={"u": lambda x: torch.ones_like(x) + torch.sin(x)},
    )
    return ph

physics_list = {b: make_physics(b) for b in BETAS}

# ── Список чекпоинтов ─────────────────────────────────────────────────────────
steps = list(range(0, MAX_STEP + 1, CHECKPOINT_STEP))
# Оставляем только существующие
steps = [s for s in steps if os.path.exists(f"{CHECKPOINT_DIR}/ckpt_{s}.pt")]
if not steps:
    raise FileNotFoundError(f"Нет чекпоинтов в {CHECKPOINT_DIR}/ckpt_*.pt")
print(f"Найдено чекпоинтов: {len(steps)}  ({steps[0]} … {steps[-1]})")

# ── Функция предсказания ──────────────────────────────────────────────────────
def predict(net, ph):
    with torch.no_grad():
        raw  = net(coords, ph.par.tensor)
        pred = ph.apply_transforms(raw)
    return pred["u"].squeeze(1).cpu().reshape(N_TIME, N_GRID).numpy()

# ── Макет: 2 строки × 3 столбца + 1 колорбар ─────────────────────────────────
N_ROWS, N_COLS = 2, 3
fig = plt.figure(figsize=(N_COLS * 3.5, N_ROWS * 3.8 + 0.6), dpi=120)

gs = gridspec.GridSpec(
    N_ROWS, N_COLS + 1,
    figure=fig,
    width_ratios=[1, 1, 1, 0.06],
    wspace=0.08,
    hspace=0.35,
    left=0.06, right=0.92, top=0.88, bottom=0.08,
)

axes = []
for r in range(N_ROWS):
    for c in range(N_COLS):
        ax = fig.add_subplot(gs[r, c])
        axes.append(ax)

cbar_ax = fig.add_subplot(gs[:, N_COLS])

# Инициализируем imshow
ims_ax = []
extent = [0, 1, 0, 2 * math.pi]

for i, (ax, beta) in enumerate(zip(axes, BETAS)):
    blank = np.zeros((N_GRID, N_TIME))
    im = ax.imshow(
        blank,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=CMAP,
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
    )
    ims_ax.append(im)

    ax.set_title(rf"$\beta = {beta}$", fontsize=10, pad=3)
    ax.set_xlabel("$t$", fontsize=9)

    row, col = divmod(i, N_COLS)
    if col == 0:
        ax.set_ylabel("$x$", fontsize=9)
        ax.set_yticks([0, math.pi, 2 * math.pi])
        ax.set_yticklabels(["$0$", "$\pi$", "$2\pi$"], fontsize=8)
    else:
        ax.set_yticks([])

    ax.set_xticks([0, 0.5, 1.0])
    ax.tick_params(labelsize=8)

# Колорбар
cb = fig.colorbar(ims_ax[0], cax=cbar_ax, label="$u$")
cb.ax.tick_params(labelsize=8)

suptitle = fig.suptitle("", fontsize=12, y=0.97, fontweight="bold")

# ── Кэш предсказаний (чтобы не грузить чекпоинт дважды) ─────────────────────
cache = {}

def get_predictions(step):
    if step in cache:
        return cache[step]
    ckpt_path = f"{CHECKPOINT_DIR}/ckpt_{step}.pt"
    net, _ = Trainer.load_checkpoint(path=ckpt_path, device=device)
    net.eval()
    preds = {}
    for beta in BETAS:
        ph = physics_list[beta]
        u_pred  = predict(net, ph)
        u_exact = exact[beta]
        l2 = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)
        preds[beta] = (u_pred, l2)
    cache[step] = preds
    return preds

# Предзагрузка всех кадров
print("Загрузка чекпоинтов и вычисление предсказаний...")
for s in steps:
    get_predictions(s)
    print(f"  ckpt_{s}.pt ✓", end="\r")
print(f"\nГотово. Генерирую GIF ({len(steps)} кадров)...")

# ── Анимация ──────────────────────────────────────────────────────────────────
def update(frame_idx):
    step  = steps[frame_idx]
    preds = cache[step]

    total_l2 = np.mean([preds[b][1] for b in BETAS])

    for i, beta in enumerate(BETAS):
        u_pred, _ = preds[beta]
        ims_ax[i].set_data(u_pred.T)

    suptitle.set_text(
        f"Шаг {step:>5d} / {MAX_STEP}    "
        rf"  $\overline{{L_2^{{rel}}}}$ = {total_l2:.4f}"
    )
    return ims_ax + [suptitle]

ani = FuncAnimation(
    fig, update,
    frames=len(steps),
    interval=1000 // FPS,
    blit=False,
)

writer = PillowWriter(fps=FPS)
ani.save(OUTPUT_GIF, writer=writer)
plt.close(fig)
print(f"Сохранено → {OUTPUT_GIF}")
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("src/")
os.makedirs("images", exist_ok=True)

from training.trainer            import Trainer
from physics.problems.elastic_wave import ElasticWave2D

# ── Настройки ────────────────────────────────────────────────────────────────
CHECKPOINT = "checkpoints/elastic/ckpt_5000.pt"
OUTPUT_PNG = "images/elastic_vnorm.png"

X_MIN, X_MAX = 0.0, 30.0
Y_MIN, Y_MAX = 0.0, 30.0

T_PLOT = 0.1
F0     = 1400/(500 * 0.15)
N      = 2000

CMAP = "magma"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Сетка (t, y, x) ──────────────────────────────────────────────────────────
xs = torch.linspace(X_MIN, X_MAX, N)
ys = torch.linspace(Y_MIN, Y_MAX, N)
YY, XX = torch.meshgrid(ys, xs, indexing="ij")
TT     = torch.full_like(XX, T_PLOT)

coords = torch.stack([TT.reshape(-1),
                      YY.reshape(-1),
                      XX.reshape(-1)], dim=1).to(device)

# ── Модель ───────────────────────────────────────────────────────────────────
net, step = Trainer.load_checkpoint(path=CHECKPOINT, device=device)
net.eval()
print(f"Шаг: {step}")

mu = torch.tensor([[F0]], dtype=torch.float32, device=device)

with torch.no_grad():
    pred = net(coords, mu)

vx = pred["vx"].squeeze(1).cpu().numpy().reshape(N, N)
vy = pred["vy"].squeeze(1).cpu().numpy().reshape(N, N)
vnorm = np.sqrt(vx ** 2 + vy ** 2)

# ── График ───────────────────────────────────────────────────────────────────
vmax = np.nanpercentile(vnorm, 99.5)

fig, ax = plt.subplots(figsize=(7.0, 6.4), dpi=1000)
im = ax.imshow(
    vnorm,
    origin="lower",
    cmap=CMAP,
    extent=[X_MIN, X_MAX, Y_MIN, Y_MAX],
    interpolation="bilinear",
    # vmin=0.0,
    # vmax=vmax,
)

cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label(r"$|\mathbf{v}|$", fontsize=13)

ax.set_aspect("equal")
ax.set_xlabel(r"$x$", fontsize=13)
ax.set_ylabel(r"$y$", fontsize=13)
ax.set_title(rf"$|\mathbf{{v}}(x,y,\,t={T_PLOT:g})|$", fontsize=13, pad=10)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, bbox_inches="tight", dpi=300)
print(f"Сохранено → {OUTPUT_PNG}")

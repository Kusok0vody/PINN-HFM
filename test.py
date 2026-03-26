import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.append("src/")

from geometry.geom       import Geometry
from geometry.sampler    import Sampler
from network.net         import Net
from network.activations import Sine
from physics.problems.proppant import proppantDynamics_dless
from pinn                import PINN
from utils               import smooth_clamp


# =============================================================================
# Параметры сетки и времени
# =============================================================================

N_grid = 128           # разрешение сетки N x N
T_slices = [0.0, 0.25, 0.5, 0.75, 1.0]   # моменты времени для срезов
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Физические параметры (из debug.py)
# =============================================================================

rho_f = 1.0
rho_p = 1.2
g     = 0.0
H     = 1;  L = 1

p0    = 1 / (12 * 0.01 * L * 1)
G     = p0 * H * rho_f * g
r     = 0.65 * (rho_p - rho_f) / rho_f
alpha = L / H

mu_list = [
    {"alpha": alpha, "beta": -2.5, "r": r, "G": G},
]


# =============================================================================
# Архитектура (должна совпадать с обученной моделью)
# =============================================================================

outputs_config = {
    "c":  {},
    "px": {},
    "py": {},
}

net = Net(
    x_dim=3,
    mu_dim=4,
    dx=32,
    dmu=32,
    d_h=64,
    encoder_layers=2,
    trunk_layers=4,
    head_layers=2,
    activation=nn.Tanh,
    trunk_activation=Sine,
    outputs_config=outputs_config,
    use_film=True,
    use_fourier=True,
    n_freqs=8,
    omega_min=1.0,
    omega_max=32.0,
)


# =============================================================================
# Загрузка чекпоинта
# =============================================================================

CHECKPOINT = "checkpoints_test/ckpt_14975.pt"

ckpt = torch.load(CHECKPOINT, map_location=device)
net.load_state_dict(ckpt["net"])
net.to(device)
net.eval()
print(f"Loaded checkpoint: {CHECKPOINT}")


# =============================================================================
# Физика для set_par
# =============================================================================

bounds = {
    "inlet":  {"p": [0.0, 1.0], "x": lambda p: 0*p,     "y": lambda p: p      },
    "outlet": {"p": [0.0, 1.0], "x": lambda p: 0*p + 1, "y": lambda p: p      },
    "bottom": {"p": [0.0, 1.0], "x": lambda p: p,       "y": lambda p: 0*p    },
    "top":    {"p": [0.0, 1.0], "x": lambda p: p,       "y": lambda p: 0*p + 1},
}

physics = proppantDynamics_dless(dim=2, has_time=True, device=device)
physics.setParameters(
    params={
        "alpha": torch.tensor([m["alpha"] for m in mu_list]).to(device),
        "beta":  torch.tensor([m["beta"]  for m in mu_list]).to(device),
        "r":     torch.tensor([m["r"]     for m in mu_list]).to(device),
        "G":     torch.tensor([m["G"]     for m in mu_list]).to(device),
    },
    funcPar={"w": lambda x, y: torch.ones_like(x)},
    boundaries=bounds,
)

mu_tensor = physics.set_par().to(device)   # (1, 4)


# =============================================================================
# Построение сетки и предсказание
# =============================================================================

xs = torch.linspace(0.0, 1.0, N_grid)
ys = torch.linspace(0.0, 1.0, N_grid)
YY, XX = torch.meshgrid(ys, xs, indexing="ij")   # (N_grid, N_grid)

XX_flat = XX.reshape(-1, 1)   # (N², 1)
YY_flat = YY.reshape(-1, 1)


def predict_at_t(t_val: float) -> dict:
    """Returns dict of (N_grid, N_grid) numpy arrays for given t."""
    T_flat  = torch.full_like(XX_flat, t_val)
    coords  = torch.cat([T_flat, YY_flat, XX_flat], dim=1).to(device)  # (N², 3) — [t, y, x]

    with torch.no_grad():
        raw  = net(coords, mu_tensor)
        pred = physics.apply_transforms(raw)

    width   = torch.ones_like(XX_flat).to(device)
    c       = pred["c"][:, 0:1]     # (N², 1) — первая постановка
    p_x     = pred["px"][:, 0:1]
    p_y     = pred["py"][:, 0:1]

    beta    = physics.par["beta"].unsqueeze(0)
    mu_visc = (1 - c) ** beta
    mob     = width**2 / mu_visc

    alpha_v = physics.par["alpha"].unsqueeze(0)
    r_v     = physics.par["r"].unsqueeze(0)
    G_v     = physics.par["G"].unsqueeze(0)
    gravity = (1 + r_v * c) * G_v

    ux = (-mob * p_x).squeeze(1).cpu().reshape(N_grid, N_grid).numpy()
    uy = (-mob * (p_y - gravity) * alpha_v).squeeze(1).cpu().reshape(N_grid, N_grid).numpy()
    c  = c.squeeze(1).cpu().reshape(N_grid, N_grid).numpy()

    return {"c": c, "ux": ux, "uy": uy}


# =============================================================================
# Отрисовка
# =============================================================================

n_t    = len(T_slices)
fields = ["c", "ux", "uy"]
titles = {"c": "Concentration $c$", "ux": "$u_x$", "py": "$u_y$"}
cmaps  = {"c": "viridis", "ux": "RdBu_r", "uy": "RdBu_r"}

fig = plt.figure(figsize=(5 * n_t, 4 * len(fields)), dpi=150)
gs  = gridspec.GridSpec(len(fields), n_t, figure=fig, hspace=0.35, wspace=0.25)

for row, field in enumerate(fields):
    preds = [predict_at_t(t) for t in T_slices]
    vmin  = min(p[field].min() for p in preds)
    vmax  = max(p[field].max() for p in preds)

    for col, (t_val, pred) in enumerate(zip(T_slices, preds)):
        ax  = fig.add_subplot(gs[row, col])
        im  = ax.imshow(
            pred[field],
            origin="lower",
            extent=[0, 1, 0, 1],
            aspect="equal",
            cmap=cmaps[field],
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"t = {t_val:.2f}", fontsize=10)
        if col == 0:
            ax.set_ylabel(titles.get(field, field), fontsize=10)
        else:
            ax.set_yticks([])
        ax.set_xticks([0, 0.5, 1])

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle("PINN solution", fontsize=14, y=1.01)
plt.savefig("pinn_results.png", bbox_inches="tight", dpi=150)
plt.show()
print("Saved to pinn_results.png")
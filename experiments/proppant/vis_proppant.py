import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))

from network.net         import Net
from physics.problems.proppant import proppantDynamics_dless
from training.trainer import Trainer

N_grid = 500
T_slices = [0.0, 0.25, 0.5, 0.75, 1.0]

torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device != 'cpu':
    torch.cuda.set_device(device)
print(device)

rho_f = 1.0
rho_p = 1.2
g     = 0.0
H     = 1;  L = 1

p0    = 1 / (12 * 0.01 * L * 1)
G     = p0 * H * rho_f * g
r     = 0.65 * (rho_p - rho_f) / rho_f
alpha = L / H

parameters = [{"alpha": alpha, "beta": 0.0, "r": r, "G": G}]

physics = proppantDynamics_dless(dim=2, has_time=True, device=device)
physics.setParameters(
    params=parameters,
    funcPar={"w": lambda x, y: torch.ones_like(x)},
    boundaries={},
)

CHECKPOINT = "checkpoints/proppant_debug/ckpt_20000.pt"

net, step = Trainer.load_checkpoint(
    path=CHECKPOINT,
    device=device,
)

xs = torch.linspace(0.0, 1.0, N_grid)
ys = torch.linspace(0.0, 1.0, N_grid)
YY, XX = torch.meshgrid(ys, xs, indexing="ij")

XX_flat = XX.reshape(-1, 1)
YY_flat = YY.reshape(-1, 1)


def predict_at_t(t_val: float) -> dict:
    T_flat  = torch.full_like(XX_flat, t_val)
    coords  = torch.cat([T_flat, YY_flat, XX_flat], dim=1).to(device)

    with torch.no_grad():
        raw  = net(coords, physics.par.tensor)
        pred = physics.apply_transforms(raw)

    width   = torch.ones_like(XX_flat).to(device)
    c       = pred["c"][:, 0:1]
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
    c  = (c*0.65).squeeze(1).cpu().reshape(N_grid, N_grid).numpy()

    return {"c": c, "ux": ux, "uy": uy}

n_t    = len(T_slices)
fields = ["c", "ux", "uy"]
titles = {"c": "Concentration $c$", "ux": "$u_x$", "py": "$u_y$"}
cmaps  = {"c": "turbo", "ux": "jet", "uy": "seismic"}
limits = {"c": [0, 0.65], "ux": [0, 1], "uy": [-2, 2]}

fig = plt.figure(figsize=(5 * n_t, 4 * len(fields)), dpi=150)
gs  = gridspec.GridSpec(len(fields), n_t, figure=fig, hspace=0.35, wspace=0.25)

for row, field in enumerate(fields):
    preds = [predict_at_t(t) for t in T_slices]
    vmin, vmax = limits[field]

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
print("Saved --> pinn_results.png")

plt.title("ux")
plt.plot(ys, preds[0]["ux"][:,0], c='tab:red', label=np.sum(preds[0]["ux"][:,0]))
plt.plot(ys, preds[0]["ux"][:,-1], c='tab:blue', label=np.sum(preds[0]["ux"][:,-1]))
plt.grid()
plt.legend()
plt.xticks([0, 0.25, 0.5, 0.75, 1.0])
plt.show()
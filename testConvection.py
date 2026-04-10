import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.append("src/")

from physics.problems.convection1D import convection1D
from training.trainer import Trainer

# Parameters of grid and time
N_grid = 200
N_time = 200

# device
torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device != 'cpu':
    torch.cuda.set_device(device)
print(device)

# Physical parameters
parameters = [{"beta": 5}]
print(parameters)

physics = convection1D(dim=1, has_time=True, device=device)
physics.setParameters(
    params=parameters,
    boundaries={},
    initial={
        "u": lambda x: torch.ones_like(x) + torch.sin(x),
    }
)

# Checkpoint loading
CHECKPOINT = "checkpoints/proppant_debug/ckpt_5000.pt"
net, step = Trainer.load_checkpoint(path=CHECKPOINT, device=device)

# Grid
xs = torch.linspace(0.0, 6.0, N_grid)
ts = torch.linspace(0.0, 1.0, N_time)
TT, XX = torch.meshgrid(ts, xs, indexing="ij")

coords = torch.cat([TT.reshape(-1, 1), XX.reshape(-1, 1)], dim=1).to(device)
with torch.no_grad():
    raw  = net(coords, physics.par.tensor)
    pred = physics.apply_transforms(raw)

u_pred = pred["u"].squeeze(1).cpu().reshape(N_time, N_grid).numpy()

# Exact solution: u(x, t) = 1 + sin(x - beta*t)
beta = parameters[0]["beta"]
u_exact = 1.0 + np.sin(XX.numpy() - beta * TT.numpy())

cmap = "rainbow"
vmin, vmax = u_exact.min(), u_exact.max()

fig = plt.figure(figsize=(10, 7), dpi=150)
gs  = gridspec.GridSpec(
    1, 3,
    figure=fig,
    width_ratios=[1, 1, 1],
    wspace=0.35,
)

titles = ["PINN prediction", "Exact solution", "Absolute error"]
data   = [u_pred, u_exact, np.abs(u_pred - u_exact)]

axes = []
ims  = []
for col, (title, arr) in enumerate(zip(titles, data)):
    ax = fig.add_subplot(gs[0, col])
    axes.append(ax)

    _vmin = 0          if col == 2 else vmin
    _vmax = vmax * 0.5 if col == 2 else vmax
    _cmap = "hot_r"    if col == 2 else cmap

    im = ax.imshow(
        arr.T,
        origin="lower",
        extent=[0, 1, 0, 6],
        aspect="auto",
        cmap=_cmap,
        vmin=_vmin,
        vmax=_vmax,
        interpolation="none",
    )
    ims.append(im)

    ax.set_title(title, fontsize=11, pad=6)
    ax.set_xlabel("$t$", fontsize=11)
    if col == 0:
        ax.set_ylabel("$x$", fontsize=11)
    else:
        ax.set_yticklabels([])

    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks(np.linspace(0, 6, 7))

fig.colorbar(ims[0], ax=[axes[0], axes[1]], label="$u$", fraction=0.046, pad=0.04, location='left')
fig.colorbar(ims[2], ax=axes[2], label="|error|", fraction=0.046, pad=0.04)

fig.suptitle(
    rf"Convection 1D  ($\beta = {beta}$, step {step})",
    fontsize=13, y=1.02
)

plt.savefig("pinn_results_convection.png", bbox_inches="tight", dpi=150)
plt.show()
print("Saved → pinn_results_convection.png")
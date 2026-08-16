import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from numerical import solve_and_plot

sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))

from physics.problems.cdr import CDR1D
from training.trainer import Trainer
from utils import unpack_coords

def u0(x):
    sigma = np.pi / 4
    return np.exp(-(x - np.pi) ** 2 / (2.0 * sigma ** 2))

def f_source(u):
    return u * (1 - u)

N_grid = 400
N_time = 800

# torchic = lambda x: torch.maximum(torch.zeros_like(x), torch.sign(torch.sin(2*x)))
# numpyic = lambda x: np.maximum(np.zeros_like(x), np.sign(np.sin(2*x)))

def numpyic(x, eps=1e-12):
    y = np.sin(2*x)
    return (y > eps).astype(float)

def torchic(x, eps=1e-12):
    y = torch.sin(2*x)
    return (y > eps).float()

torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device != 'cpu':
    torch.cuda.set_device(device)
print(device)

beta = 2.0
nu = 0.1
rho = 2.0
parameters = [{"beta": beta, "nu": nu, "rho": rho}]
print(parameters)

physics = CDR1D(dim=1, has_time=True, device=device)
physics.setParameters(
    params=parameters,
    boundaries={},
    # initial={
    #     "u": torchic,
    # },
    hard_ic=True
)

CHECKPOINT = "checkpoints/cdr/ckpt_30100.pt"
net, step = Trainer.load_checkpoint(path=CHECKPOINT, device=device)

xmax = 2*torch.pi
tmax = 1.0
xs = torch.linspace(0.0, xmax, N_grid)
ts = torch.linspace(0.0, tmax, N_time)
TT, XX = torch.meshgrid(ts, xs, indexing="ij")

coords = torch.cat([TT.reshape(-1, 1), XX.reshape(-1, 1)], dim=1).to(device)
unpacked, _ = unpack_coords(coords, has_time=True, dim=1)

with torch.no_grad():
    raw  = net(coords, physics.par.tensor)
    pred = physics.apply_transforms(raw)
    pred = physics.apply_output_ansatz(pred, unpacked)

u_pred = pred["u"].squeeze(1).cpu().reshape(N_time, N_grid).numpy()

x, t, u_exact = solve_and_plot(beta, nu, rho, xmax, tmax, N_grid, N_time, u0, f_source, limiter='vanleer')

u_pred_flat = u_pred.flatten()
u_exact_flat = u_exact.flatten()
l2_rel = np.linalg.norm(u_pred_flat - u_exact_flat) / np.linalg.norm(u_exact_flat)
print(f"L2 relative error: {l2_rel:.4f}")

cmap = "rainbow"
vmin, vmax = 0, 2

fig = plt.figure(figsize=(10, 7), dpi=300)
gs = gridspec.GridSpec(
    1, 5,
    figure=fig,
    width_ratios=[0.05, 1, 1, 1, 0.05],
    wspace=0.35,
)

# titles = ["PINN prediction", "numerical solution", "Absolute error"]
titles = ["Предсказание PINN", "Численное решение", "Абсолютная ошибка"]
data   = [u_pred, u_exact, np.abs(u_pred - u_exact)]

cax_left  = fig.add_subplot(gs[0, 0])
ax0       = fig.add_subplot(gs[0, 1])
ax1       = fig.add_subplot(gs[0, 2])
ax2       = fig.add_subplot(gs[0, 3])
cax_right = fig.add_subplot(gs[0, 4])

pos = cax_left.get_position()

cax_left.set_position([
    pos.x0 - 0.02,
    pos.y0,
    pos.width,
    pos.height
])

xticks = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
xticklabels = [r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"]

axes = [ax0, ax1, ax2]
ims  = []
for col, (ax, title, arr) in enumerate(zip(axes, titles, data)):
    # ax = fig.add_subplot(gs[0, col])
    # axes.append(ax)

    _vmin = 0          if col == 2 else vmin
    _vmax = vmax * 0.5 if col == 2 else vmax
    _cmap = "hot_r"    if col == 2 else cmap

    im = ax.imshow(
        arr.T,
        origin="lower",
        extent=[0, tmax, 0, xmax],
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
        ax.set_yticklabels([
            r"$0$",
            r"$\frac{1}{2}\pi$",
            r"$\pi$",
            r"$\frac{3}{2}\pi$",
            r"$2\pi$"
        ])
    else:
        ax.set_yticklabels([])

    ax.set_xticks(np.linspace(0, tmax, 5))
    ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])

fig.colorbar(ims[0], cax=cax_left,  label=r"$u$")
cax_left.yaxis.set_ticks_position("left")
cax_left.yaxis.set_label_position("left")

fig.colorbar(ims[2], cax=cax_right, label=r"$|u_{\mathrm{PINN}} - u_{\mathrm{num}}|$")

fig.suptitle(
    rf"Конвекция-диффузия-реакция 1D ($\beta = {beta}$, $\nu = {nu}$, $\rho = {rho}$ step {step})",
    # rf"CDR 1D ($\beta = {beta}$, $\nu = {nu}$, $\rho = {rho}$ step {step})",
    fontsize=13, y=1.02
)

plt.savefig("images/pinn_results_cdr.png", bbox_inches="tight", dpi=300)
plt.savefig("images/pinn_results_cdr.pdf", bbox_inches="tight", dpi=200)
plt.show()
print("Saved --> images/pinn_results_cdr.png")
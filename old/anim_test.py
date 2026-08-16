import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.append("src/")

from physics.problems.proppant import proppantDynamics_dless
from training.trainer import Trainer

# Config
N_grid    = 200
N_frames  = 100
T_min     = 0.0
T_max     = 1.0
CHECKPOINT = "checkpoints/proppant_debug/ckpt_50000.pt"
OUTPUT_GIF = "pinn_results.gif"
FIELDS     = ["c", "ux", "uy"]
CMAPS      = {"c": "turbo", "ux": "jet", "uy": "seismic"}
TITLES     = {"c": "Concentration $c$", "ux": "$u_x$", "uy": "$u_y$"}
LIMITS     = {"c": [0, 1], "ux": [0, 1], "uy": [-2, 2]}

# Device 
torch.manual_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Physics 
rho_f = 1.0;  rho_p = 1.2;  g = 0.0
H = 1;        L = 1
p0    = 1 / (12 * 0.01 * L * 1)
G     = p0 * H * rho_f * g
r     = 0.65 * (rho_p - rho_f) / rho_f
alpha = L / H

physics = proppantDynamics_dless(dim=2, has_time=True, device=device)
physics.setParameters(
    params=[{"alpha": alpha, "beta": 0.0, "r": r, "G": G}],
    funcPar={"w": lambda x, y: torch.ones_like(x)},
    boundaries={},
)

# Network
net, step = Trainer.load_checkpoint(path=CHECKPOINT, device=device)
net.eval()
print(f"Loaded checkpoint at step {step}")

# Grid
xs = torch.linspace(0.0, 1.0, N_grid)
ys = torch.linspace(0.0, 1.0, N_grid)
YY, XX = torch.meshgrid(ys, xs, indexing="ij")
XX_flat = XX.reshape(-1, 1)
YY_flat = YY.reshape(-1, 1)

# Predict 
def predict_at_t(t_val: float) -> dict:
    T_flat = torch.full_like(XX_flat, t_val)
    coords = torch.cat([T_flat, YY_flat, XX_flat], dim=1).to(device)

    with torch.no_grad():
        raw  = net(coords, physics.par.tensor)
        pred = physics.apply_transforms(raw)

    width   = torch.ones_like(XX_flat).to(device)
    c       = pred["c"][:, 0:1]
    p_x     = pred["px"][:, 0:1]
    p_y     = pred["py"][:, 0:1]

    beta    = physics.par["beta"].unsqueeze(0)
    mu_visc = (1 - c) ** beta
    mob     = width ** 2 / mu_visc

    alpha_v = physics.par["alpha"].unsqueeze(0)
    r_v     = physics.par["r"].unsqueeze(0)
    G_v     = physics.par["G"].unsqueeze(0)
    gravity = (1 + r_v * c) * G_v

    return {
        "c":  c.squeeze(1).cpu().reshape(N_grid, N_grid).numpy(),
        "ux": (-mob * p_x).squeeze(1).cpu().reshape(N_grid, N_grid).numpy(),
        "uy": (-mob * (p_y - gravity) * alpha_v).squeeze(1).cpu().reshape(N_grid, N_grid).numpy(),
    }

# precompute all frames
t_vals = np.linspace(T_min, T_max, N_frames)
print("Precomputing frames...")
frames = [predict_at_t(t) for t in t_vals]
print("Done.")

# global vmin/vmax per field for consistent colorbar
# vlims = {
#     field: (
#         min(f[field].min() for f in frames),
#         max(f[field].max() for f in frames),
#     )
#     for field in FIELDS
# }

# Animation
n_fields = len(FIELDS)
fig, axes = plt.subplots(1, n_fields, figsize=(5 * n_fields, 4.5), dpi=120)

ims = []
for ax, field in zip(axes, FIELDS):
    vmin, vmax = LIMITS[field]
    im = ax.imshow(
        frames[0][field],
        origin="lower",
        extent=[0, 1, 0, 1],
        aspect="equal",
        cmap=CMAPS[field],
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(TITLES[field], fontsize=11)
    ax.set_xlabel("x");  ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ims.append(im)

time_text = fig.suptitle("t = 0.000", fontsize=13)
fig.tight_layout()

def update(i):
    for im, field in zip(ims, FIELDS):
        im.set_data(frames[i][field])
    time_text.set_text(f"t = {t_vals[i]:.3f}")
    return ims + [time_text]

ani = animation.FuncAnimation(
    fig, update, frames=N_frames, interval=100, blit=True
)

ani.save(OUTPUT_GIF, writer="pillow", fps=10)
print(f"Saved to {OUTPUT_GIF}")
plt.show()
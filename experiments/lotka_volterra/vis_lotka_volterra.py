import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))
os.makedirs("images", exist_ok=True)

from physics.problems.lotka_volterra import LotkaVolterra
from training.trainer                import Trainer

CHECKPOINT = "checkpoints/lotka_volterra/ckpt_24000.pt"
OUTPUT_PNG = "images/lotka_volterra.png"

T_END = 30.0
N_EVAL = 1000

PARAMS = {
    "r":     1.0,
    "K":     1.0,
    "alpha": 1.0,
    "e1":    0.5,
    "d1":    0.3,
    "beta":  0.8,
    "e2":    0.6,
    "d2":    0.2,
}

G0 = 0.8
H0 = 0.4
P0 = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net, step = Trainer.load_checkpoint(path=CHECKPOINT, device=device)
net.eval()
print(f"Step: {step}")

physics = LotkaVolterra(dim=1, has_time=False, device=device)
physics.setParameters(params=[PARAMS], boundaries={}, initial=None)

t_vals = torch.linspace(0.0, T_END, N_EVAL).unsqueeze(1).to(device)

with torch.no_grad():
    raw  = net(t_vals, physics.par.tensor)
    pred = physics.apply_transforms(raw)

t_np = t_vals.cpu().squeeze().numpy()
G_pinn = pred["G"].squeeze(1).cpu().numpy()
H_pinn = pred["H"].squeeze(1).cpu().numpy()
P_pinn = pred["P"].squeeze(1).cpu().numpy()

try:
    from scipy.integrate import solve_ivp

    p = PARAMS
    def rhs(t, y):
        G, H, P = y
        dG = p["r"] * G * (1 - G / p["K"]) - p["alpha"] * G * H
        dH = p["e1"] * p["alpha"] * G * H - p["d1"] * H - p["beta"] * H * P
        dP = p["e2"] * p["beta"] * H * P - p["d2"] * P
        return [dG, dH, dP]

    sol = solve_ivp(rhs, [0, T_END], [G0, H0, P0],
                    t_eval=t_np, method="RK45", rtol=1e-8, atol=1e-10)
    G_ref = sol.y[0]
    H_ref = sol.y[1]
    P_ref = sol.y[2]
    has_ref = True
except ImportError:
    has_ref = False
    print("scipy not found, skipping reference solution")

mpl.rcParams.update({
    "font.size":      13,
    "axes.labelsize": 14,
    "font.family":    "Montserrat",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)

ax.plot(t_np, G_pinn, color="#2ca02c", linewidth=5.0, zorder=2, label="$G(t)$ -- grass (PINN)")
ax.plot(t_np, H_pinn, color="#1f77b4", linewidth=5.0, zorder=2, label="$H(t)$ -- herbivores (PINN)")
ax.plot(t_np, P_pinn, color="#d62728", linewidth=5.0, zorder=2, label="$P(t)$ -- predators (PINN)")

if has_ref:
    ax.plot(t_np, G_ref, color="#1a6b1a", linewidth=2.0,
            linestyle="--", zorder=3, label="$G(t)$ -- numerical")
    ax.plot(t_np, H_ref, color="#104f8a", linewidth=2.0,
            linestyle="--", zorder=3, label="$H(t)$ -- numerical")
    ax.plot(t_np, P_ref, color="#8b1010", linewidth=2.0,
            linestyle="--", zorder=3, label="$P(t)$ -- numerical")

ax.set_xlabel("$t$")
ax.set_ylabel("Численность популяции")
ax.legend(fontsize=10, ncol=2 if has_ref else 1,
          framealpha=0.9)
ax.set_xlim(0, T_END)
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, bbox_inches="tight", dpi=300)
print(f"Saved --> {OUTPUT_PNG}")



def plot_lotka_volterra(net, physics, T_end, t_np=None, ax=None, lw=2.0):
    device_net = next(net.parameters()).device

    if t_np is None:
        t_np = np.linspace(0.0, T_end, 1000)

    t_t = torch.tensor(t_np, dtype=torch.float32).unsqueeze(1).to(device_net)

    with torch.no_grad():
        raw  = net(t_t, physics.par.tensor)
        pred = physics.apply_transforms(raw)

    G = pred["G"].squeeze(1).cpu().numpy()
    H = pred["H"].squeeze(1).cpu().numpy()
    P = pred["P"].squeeze(1).cpu().numpy()

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(t_np, G, color="#2ca02c", linewidth=lw, label="$G(t)$")
    ax.plot(t_np, H, color="#1f77b4", linewidth=lw, label="$H(t)$")
    ax.plot(t_np, P, color="#d62728", linewidth=lw, label="$P(t)$")

    ax.set_xlabel("$t$")
    ax.set_ylabel("Population size")
    ax.set_xlim(0, T_end)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=11)

    return ax

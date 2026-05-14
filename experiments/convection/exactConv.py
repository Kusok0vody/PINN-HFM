import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

os.makedirs("images", exist_ok=True)

BETA = 15
N    = 600

x = np.linspace(0, 2 * np.pi, N)
t = np.linspace(0, 1, N)
tt, xx = np.meshgrid(t, x, indexing="ij")
u = 1.0 + np.sin(xx - BETA * tt)

mpl.rcParams.update({
    "font.size":        13,
    "axes.labelsize":   14,
    "axes.titlesize":   15,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "font.family":      "DejaVu Sans",
})

fig, ax = plt.subplots(figsize=(7, 5), dpi=200)

im = ax.imshow(
    u.T,
    cmap="rainbow",
    extent=[0, 1, 0, 2 * np.pi],
    origin="lower",
    aspect="auto",
    interpolation="bilinear",
    vmin=0.0,
    vmax=2.0,
)

cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
cb.set_label(r"$u(x,\,t)$", fontsize=13)
cb.ax.tick_params(labelsize=10)

ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$x$")
ax.set_yticks([0, np.pi, 2 * np.pi])
ax.set_yticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])

ax.set_title(
    rf"Exact solution of 1D convection, $\beta = {BETA}$",
    pad=10,
)

plt.tight_layout()
plt.savefig("images/exact.png", bbox_inches="tight", dpi=300)
print("Saved --> images/exact.png")

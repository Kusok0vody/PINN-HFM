import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches


def plot_samples(pts, title="", dpi=300, t_fixed=None, xlim=None, ylim=None):
    fig, ax = plt.subplots(dpi=dpi)
    ax.set_title(title, fontsize=13)
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.4, alpha=0.5)

    c = pts.interior.coords
    ax.scatter(c[:, -1].numpy(), c[:, -2].numpy() if c.shape[1] >= 2 else c[:, -1].numpy(),
               s=5, color="tab:red", alpha=1.0, edgecolors="black", linewidths=1, label="interior", zorder=2)

    for name, b in pts.boundaries.items():
        xb = b.coords[:, -1].numpy()
        yb = b.coords[:, -2].numpy() if b.coords.shape[1] >= 2 else b.coords[:, -1].numpy()
        ax.scatter(xb, yb, s=18, color="tab:green", edgecolors="black", linewidths=1, zorder=4,
                   label="boundary" if name == list(pts.boundaries)[0] else "")
        nx = b.nx[:, 0].numpy()
        ny = b.ny[:, 0].numpy()
        scale = 0.06
        ax.quiver(xb, yb, nx, ny, color="tab:purple", scale=1/scale,
                  scale_units="xy", width=0.003, alpha=1.0,
                  label="normals" if name == list(pts.boundaries)[0] else "")

    if pts.initial is not None:
        ci = pts.initial.coords
        ax.scatter(ci[:, -1].numpy(), ci[:, -2].numpy() if ci.shape[1] >= 2 else ci[:, -1].numpy(),
                   s=10, color="tab:blue", alpha=1.0, edgecolors="black", linewidths=1,
                   label="initial (t=0)", zorder=3)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.legend(fontsize=9, markerscale=1.5)
    plt.tight_layout()
    plt.show()


def plot_samples_3d(pts, title="", dpi=300, xlim=None, ylim=None, zlim=None):
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title, fontsize=13)

    c = pts.interior.coords.numpy()
    ax.scatter(c[:, 0], c[:, 2], c[:, 1], s=5, color="tab:red", alpha=1.0,
               edgecolors="black", linewidths=1, label="interior")

    for b in pts.boundaries.values():
        bc = b.coords.numpy()
        ax.scatter(bc[:, 0], bc[:, 2], bc[:, 1], s=12, color="tab:green",
                   edgecolors="black", linewidths=1, zorder=4)

    if pts.initial is not None:
        ic = pts.initial.coords.numpy()
        ax.scatter(ic[:, 0], ic[:, 2], ic[:, 1], s=12, color="tab:blue",
                   edgecolors="black", linewidths=1, label="initial")

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if zlim is not None:
        ax.set_zlim(zlim)

    ax.set_xlabel("t"); ax.set_ylabel("x"); ax.set_zlabel("y")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()
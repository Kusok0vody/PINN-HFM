import torch
import numpy as np
import matplotlib.pyplot as plt


def make_grid_2d(x_range, y_range, N, t=None, device="cpu"):
    """
    Build an evaluation grid for a 2D field.

    Returns (coords, XX, YY):
        coords — (N*N, n_coords) tensor, columns ordered (t, y, x) if t is not
                 None else (y, x); ready to feed into Net
        XX, YY — (N, N) numpy meshgrid arrays for plotting
    """
    xs = torch.linspace(x_range[0], x_range[1], N)
    ys = torch.linspace(y_range[0], y_range[1], N)
    YY, XX = torch.meshgrid(ys, xs, indexing="ij")

    cols = [YY.reshape(-1, 1), XX.reshape(-1, 1)]
    if t is not None:
        cols = [torch.full_like(XX.reshape(-1, 1), t)] + cols

    coords = torch.cat(cols, dim=1).to(device)
    return coords, XX.numpy(), YY.numpy()


def eval_field(net, coords, mu, key, N):
    """Evaluate net and return field `key` as an (N, N) numpy array."""
    with torch.no_grad():
        pred = net(coords, mu)
    return pred[key].squeeze(1).cpu().numpy().reshape(N, N)


def eval_velocity_norm(net, coords, mu, N, keys=("vx", "vy")):
    """Evaluate net and return sqrt(vx^2 + vy^2) as an (N, N) numpy array."""
    with torch.no_grad():
        pred = net(coords, mu)
    vx = pred[keys[0]].squeeze(1).cpu().numpy().reshape(N, N)
    vy = pred[keys[1]].squeeze(1).cpu().numpy().reshape(N, N)
    return np.sqrt(vx ** 2 + vy ** 2)


def plot_field(values, x_range, y_range, title="", label="", cmap="magma",
               mask=None, symmetric=False, vmax=None, contours=0,
               figsize=(7.0, 6.4), dpi=200, output_path=None):
    """
    Heatmap of a scalar field on a rectangle.

    Args:
        values     — (N, N) array
        x_range,
        y_range    — (min, max) extents
        symmetric  — center the colormap at 0 (for signed fields)
        vmax       — explicit colour limit; if None, uses the 99.5th percentile
        mask       — optional boolean (N, N) array; False cells set to NaN
        contours   — number of white contour lines (0 to disable)
        output_path — if given, save the figure instead of returning

    Returns (fig, ax) when output_path is None.
    """
    field = np.where(mask, values, np.nan) if mask is not None else values

    if vmax is None:
        vmax = np.nanpercentile(np.abs(field), 99.5)
    vmin = -vmax if symmetric else 0.0 if not symmetric and field.min() >= 0 else -vmax

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(
        field,
        origin="lower",
        cmap=cmap,
        extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
        interpolation="bilinear",
        vmin=vmin,
        vmax=vmax,
    )

    if contours > 0:
        xs = np.linspace(x_range[0], x_range[1], field.shape[1])
        ys = np.linspace(y_range[0], y_range[1], field.shape[0])
        ax.contour(xs, ys, field, levels=contours, colors="white",
                   linewidths=0.55, alpha=0.55)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(label, fontsize=13)

    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$", fontsize=13)
    ax.set_ylabel(r"$y$", fontsize=13)
    if title:
        ax.set_title(title, fontsize=13, pad=10)

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        return None
    return fig, ax

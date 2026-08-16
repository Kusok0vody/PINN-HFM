"""
GIF: предсказание PINN для всех постановок из объекта физики.

Функция make_param_sweep_gif — универсальная: принимает любой объект физики,
проходит по physics.par.tensor и рисует по одному кадру на постановку.

Запуск: python vis_convection_sweep.py
Требует: checkpoints/convection/ckpt_20000.pt
"""

import sys
import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.append("src/")
os.makedirs("gifs", exist_ok=True)

from physics.problems.convection1D import convection1D
from training.trainer import Trainer


# ══════════════════════════════════════════════════════════════════════════════
#  Универсальная функция
# ══════════════════════════════════════════════════════════════════════════════

def make_param_sweep_gif(
    net,
    physics,
    coords,
    output_shape,
    extent,
    output_path,
    title_fn,
    exact_fn=None,
    fps=4,
    cmap_pred="rainbow",
    cmap_err="hot_r",
    vmin=None,
    vmax=None,
    err_vmax=None,
    xlabel="",
    ylabel="",
    ytick_vals=None,
    ytick_labels=None,
):
    """
    Создаёт GIF, проходя по всем постановкам (строкам physics.par.tensor).

    Parameters
    ----------
    net          : обученная сеть
    physics      : объект Physics с заполненным par.tensor (N_params × n_dims)
    coords       : тензор входных координат (N_points × coord_dim), на нужном device
    output_shape : форма выходного массива, напр. (N_TIME, N_GRID)
    extent       : [xmin, xmax, ymin, ymax] для imshow
    output_path  : путь сохранения GIF
    title_fn     : callable(param_dict: dict[str, float]) -> str — заголовок кадра
    exact_fn     : callable(param_dict) -> np.ndarray[output_shape] или None.
                   Если задана — показываются 3 панели (PINN | Точное | Ошибка),
                   иначе — 1 панель (PINN).
    fps          : кадров в секунду
    cmap_pred    : colormap для решения
    cmap_err     : colormap для ошибки
    vmin, vmax   : пределы цветовой шкалы решения (None = авто по первому кадру)
    err_vmax     : верхний предел шкалы ошибки (None = авто)
    xlabel, ylabel : подписи осей
    ytick_vals   : позиции меток оси Y (опционально)
    ytick_labels : текстовые метки оси Y (опционально)
    """

    par_tensor   = physics.par.tensor          # (N, n_dims)
    param_names  = physics.par.names           # ["beta"] или ["k"] и т.д.
    N_params     = par_tensor.shape[0]
    device       = next(net.parameters()).device

    # ── Предвычисление всех кадров ────────────────────────────────────────────
    frames = []
    for i in range(N_params):
        row  = par_tensor[i : i + 1].to(device)   # (1, n_dims)
        pdict = {name: par_tensor[i, j].item()
                 for j, name in enumerate(param_names)}

        with torch.no_grad():
            raw  = net(coords, row)
            pred = physics.apply_transforms(raw)

        u_pred = pred["u"].squeeze(1).cpu().numpy().reshape(output_shape)

        if exact_fn is not None:
            u_exact = exact_fn(pdict).reshape(output_shape)
            u_err   = np.abs(u_pred - u_exact)
            l2_rel  = (np.linalg.norm(u_pred - u_exact)
                       / (np.linalg.norm(u_exact) + 1e-12))
        else:
            u_exact = None
            u_err   = None
            l2_rel  = None

        frames.append((pdict, u_pred, u_exact, u_err, l2_rel))

    # Авто-пределы по первому кадру
    if vmin is None:
        vmin = min(f[1].min() for f in frames)
    if vmax is None:
        vmax = max(f[1].max() for f in frames)
    if err_vmax is None and exact_fn is not None:
        err_vmax = max(f[3].max() for f in frames) * 0.9

    # ── Компоновка фигуры ─────────────────────────────────────────────────────
    has_exact = exact_fn is not None

    if has_exact:
        fig = plt.figure(figsize=(12, 4.5), dpi=130)
        gs = gridspec.GridSpec(
            1, 6,
            figure=fig,
            width_ratios=[0.045, 0.06, 1, 1, 1, 0.045],
            wspace=0.10,
            left=0.06, right=0.95, top=0.82, bottom=0.13,
        )
        cb_ax    = fig.add_subplot(gs[0, 0])  # колорбар решения (слева)
        # gs[0, 1] — распорка, не назначаем
        ax_pred  = fig.add_subplot(gs[0, 2])
        ax_exact = fig.add_subplot(gs[0, 3])
        ax_err   = fig.add_subplot(gs[0, 4])
        cb_err_ax= fig.add_subplot(gs[0, 5])
    else:
        fig = plt.figure(figsize=(5.5, 4.5), dpi=130)
        gs = gridspec.GridSpec(
            1, 3,
            figure=fig,
            width_ratios=[0.06, 0.06, 1],
            wspace=0.08,
            left=0.08, right=0.95, top=0.82, bottom=0.13,
        )
        cb_ax   = fig.add_subplot(gs[0, 0])  # колорбар решения (слева)
        # gs[0, 1] — распорка, не назначаем
        ax_pred = fig.add_subplot(gs[0, 2])

    blank = np.zeros(output_shape)

    # Стиль осей
    def style_ax(ax, title):
        ax.set_facecolor("white")
        ax.set_title(title, fontsize=11, pad=5, color="black")
        ax.set_xlabel(xlabel, fontsize=10, color="black")
        ax.tick_params(colors="black", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")

    im_pred = ax_pred.imshow(
        blank, origin="lower", extent=extent, aspect="auto",
        cmap=cmap_pred, vmin=vmin, vmax=vmax, interpolation="bilinear",
    )
    style_ax(ax_pred, "Предсказание PINN")
    ax_pred.set_ylabel(ylabel, fontsize=10, color="black")
    if ytick_vals is not None:
        ax_pred.set_yticks(ytick_vals)
    if ytick_labels is not None:
        ax_pred.set_yticklabels(ytick_labels, fontsize=8)

    cb = fig.colorbar(im_pred, cax=cb_ax, label="$u$")
    cb.ax.yaxis.set_ticks_position("left")
    cb.ax.yaxis.set_label_position("left")
    cb.ax.tick_params(labelsize=8)

    if has_exact:
        im_exact = ax_exact.imshow(
            blank, origin="lower", extent=extent, aspect="auto",
            cmap=cmap_pred, vmin=vmin, vmax=vmax, interpolation="bilinear",
        )
        style_ax(ax_exact, "Точное решение")
        ax_exact.set_yticks([])

        im_err = ax_err.imshow(
            blank, origin="lower", extent=extent, aspect="auto",
            cmap=cmap_err, vmin=0, vmax=err_vmax, interpolation="bilinear",
        )
        style_ax(ax_err, "Модуль ошибки")
        ax_err.set_yticks([])

        cb_err = fig.colorbar(im_err, cax=cb_err_ax)
        cb_err.ax.tick_params(labelsize=8)

    suptitle = fig.suptitle("", fontsize=13, fontweight="bold",
                            y=0.97, color="black")
    l2_text = fig.text(
        0.5, 0.01, "",
        ha="center", va="bottom", fontsize=9, color="#555555",
        transform=fig.transFigure,
    )

    # ── Функция обновления ────────────────────────────────────────────────────
    def update(idx):
        pdict, u_pred, u_exact, u_err, l2_rel = frames[idx]

        im_pred.set_data(u_pred.T)
        suptitle.set_text(title_fn(pdict))

        if has_exact:
            im_exact.set_data(u_exact.T)
            im_err.set_data(u_err.T)
            l2_text.set_text(rf"$L_2^{{rel}}$ = {l2_rel:.4f}")
            return [im_pred, im_exact, im_err, suptitle, l2_text]

        return [im_pred, suptitle]

    ani = FuncAnimation(
        fig, update,
        frames=N_params,
        interval=1000 // fps,
        blit=False,
    )

    ani.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Sokhraneno: {output_path}  ({N_params} kadrov)")


# ══════════════════════════════════════════════════════════════════════════════
#  Конфигурация для 1D конвекции
# ══════════════════════════════════════════════════════════════════════════════

CHECKPOINT = "checkpoints/convection/ckpt_20000.pt"
OUTPUT_GIF = "gifs/convection_param_sweep.gif"
FPS        = 15

N_GRID = 180
N_TIME = 180

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type != "cpu":
    torch.cuda.set_device(device)
print(f"Device: {device}")

# Сетка координат
xs = torch.linspace(0.0, 2 * math.pi, N_GRID)
ts = torch.linspace(0.0, 1.0, N_TIME)
TT, XX = torch.meshgrid(ts, xs, indexing="ij")
coords = torch.cat([TT.reshape(-1, 1), XX.reshape(-1, 1)], dim=1).to(device)
TT_np  = TT.numpy()
XX_np  = XX.numpy()

# Загрузка чекпоинта
print(f"Zagruzka {CHECKPOINT}...")
net, step = Trainer.load_checkpoint(path=CHECKPOINT, device=device)
net.eval()
print(f"Shag: {step}")

# Физика: задаём те же beta, что использовались при обучении
BETA_MIN, BETA_MAX, N_BETA = 1.0, 20.0, 201
betas = torch.linspace(BETA_MIN, BETA_MAX, N_BETA)
parameters = [{"beta": b.item()} for b in betas]

physics = convection1D(dim=1, has_time=True, device=device)
physics.setParameters(
    params=parameters,
    boundaries={},
    initial={"u": lambda x: torch.ones_like(x) + torch.sin(x)},
)

# Точное решение для конвекции
def exact_fn(param_dict):
    beta = param_dict["beta"]
    return (1.0 + np.sin(XX_np - beta * TT_np)).astype(np.float32)

# Заголовок кадра
def title_fn(param_dict):
    return rf"1D конвекция   $\beta = {param_dict['beta']:.2f}$   |   шаг {step}"

# Запуск
make_param_sweep_gif(
    net=net,
    physics=physics,
    coords=coords,
    output_shape=(N_TIME, N_GRID),
    extent=[0, 1, 0, 2 * math.pi],
    output_path=OUTPUT_GIF,
    title_fn=title_fn,
    exact_fn=exact_fn,
    fps=FPS,
    cmap_pred="rainbow",
    cmap_err="hot_r",
    vmin=0.0,
    vmax=2.0,
    err_vmax=0.5,
    xlabel="$t$",
    ylabel="$x$",
    ytick_vals=[0, math.pi, 2 * math.pi],
    ytick_labels=["$0$", r"$\pi$", r"$2\pi$"],
)

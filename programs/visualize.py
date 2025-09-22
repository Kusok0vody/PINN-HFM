import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

import programs.misc as misc

from matplotlib import gridspec
from pathlib import Path

def plot_results(data,
                 N: int,
                 limits: list,
                 path: str = '',
                 cmaps: list = None,
                 title: list = None,
                 lims: list = None,
                 cbar_labels: list = None,
                 tick_step: float = 0.25
                 ):
    """
    Plots data from Neural Network

    Parameters:
    -----------
    data : list[np.ndarray]
        List of M arrays, each shaped (N, H, W)
    N : int
        Number of rows in the plot grid
    limits : list[float]
        Plotting area limits [x_min, x_max, y_min, y_max]
    path : str
        Path to save the figure
    cmaps : list[str]
        List of colormaps per column (length M)
    title : list[list[str]]
        Titles for each subplot, shape (M, N)
    lims : list[list[float]]
        List of [vmin, vmax] per column (length M)
    cbar_labels : list[str]
        Labels for colorbars (length M)
    tick_step : float
        Tick step on both axes
    """

    x_min, x_max, y_min, y_max = limits
    M = len(data)

    if cmaps is None:
        cmaps = ['turbo'] * M

    if cbar_labels is None:
        cbar_labels = [f'$f_{i}$' for i in range(1, M + 1)]

    if title is None:
        title = [[f'$f_{col}(x,y)$' for _ in range(N)] for col in range(M)]

    if lims is None:
        lims = [[np.min(data[col]), np.max(data[col])] for col in range(M)]

    height_ratios = []
    for i in range(2 * N - 1):
        if i % 2 == 0:
            height_ratios.append(1)
        elif i == 2 * N - 3:
            height_ratios.append(0.5 * N / (N + 1) + 0.5)
        else:
            height_ratios.append(0.5 * N / (N + 1))

    width_inches = M * 3
    height_inches = 3 * N + 0.5
    fig = plt.figure(figsize=(width_inches, height_inches), dpi=300)

    gs = gridspec.GridSpec(
        nrows=2 * N - 1,
        ncols=M,
        height_ratios=height_ratios,
        hspace=0.0,
        wspace=0.4
    )

    im_list = []
    axes = [[None] * M for _ in range(N)]

    for row in range(N):
        for col in range(M):
            ax = fig.add_subplot(gs[row * 2, col])
            im = ax.imshow(data[col][row], cmap=cmaps[col],
                           vmin=lims[col][0], vmax=lims[col][1],
                           extent=[x_min, x_max, y_max, y_min])
            ax.set_title(title[col][row], fontsize=9, pad=6)
            ax.invert_yaxis()
            ax.set_xlabel('x, м', fontsize=8)
            ax.set_ylabel('y, м', fontsize=8)
            ax.set_xticks(np.arange(x_min, x_max + tick_step, tick_step))
            ax.set_yticks(np.arange(y_min, y_max + tick_step, tick_step))
            ax.tick_params(axis='both', which='major', labelsize=5)
            ax.grid(color='lightgrey', alpha=0.5, linewidth=0.5)
            axes[row][col] = ax
            im_list.append(im)

    for col in range(M):
        ax_last = axes[N - 1][col]
        ax_prev = axes[N - 2][col]

        pos_last = ax_last.get_position()
        pos_prev = ax_prev.get_position()

        center_y = (pos_last.y0 + pos_prev.y1) / 2
        height = 0.012

        cax = fig.add_axes([pos_last.x0, center_y - height / 2, pos_last.width, height])
        cbar = fig.colorbar(im_list[(N - 1) * M + col], cax=cax, orientation='horizontal')
        cax.set_xlabel(cbar_labels[col], fontsize=8, labelpad=2)
        cax.tick_params(labelsize=6)

    plt.show()

    if path:
        fig.savefig(path, bbox_inches='tight')


def plot(data:np.ndarray, limits:list=False, title:str=None, clim:list=None):
    """
    Plots data
    
    Parameters:
    -----------
    data : np.ndarray
    limits : list[np.float64]
    Limits of plot
    title : str
    Title for a plot
    """
    plt.gca().invert_yaxis()
    if limits!=False:
        plt.imshow(data, cmap='turbo', extent=[limits[0], limits[1], limits[2], limits[3]])
    else:
        plt.imshow(data, cmap='turbo')
    plt.colorbar()
    if title!=None:
        plt.title(title)
    if clim!=None:
        plt.clim(clim[0],clim[1])


def plot_BC(data_tb:np.ndarray, data_lr:np.ndarray, x:np.ndarray, y:np.ndarray):
    """
    Plots boundaries with specified types (Neumann is 1 and Dirichlet is any other value) 
    
    Parameters:
    -----------
    data_tb : np.ndarray
    data_lr : np.ndarray
    x : np.ndarray
    Array of points of x axis
    x : np.ndarray
    Array of points of y axis
    dx : np.float64
    Step between x points
    dy : np.float64
    Step between y points
    BC_types : list
    Array of conditions
    """
    plt.figure(figsize=(7, 7))
    
    plt.subplot(221)
    plt.plot(x, data_tb[0], c='black')
    plt.grid()
    plt.title('Top')
    
    plt.subplot(222)
    plt.plot(x, data_tb[-1], c='black')
    plt.grid()
    plt.title('Bottom')
    
    plt.subplot(223)
    plt.plot(y, data_lr[:,0], c='black')
    plt.grid()
    plt.title('Left')
    
    plt.subplot(224)
    plt.plot(y, data_lr[:,-1], c='black')
    plt.grid()
    plt.title('Right')

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def check_PDE_points(t, net, Nx, Ny, eps, size):

    x = torch.linspace(0, 1, Nx)
    y = torch.linspace(0, 1, Ny)
    mesh_XYT = torch.stack(torch.meshgrid(x, y, torch.Tensor([t]).to(net.device), indexing='ij')).reshape(3, -1).T
    
    X = torch.autograd.Variable(mesh_XYT[:,0], requires_grad=True)
    Y = torch.autograd.Variable(mesh_XYT[:,1], requires_grad=True)
    T = torch.autograd.Variable(mesh_XYT[:,2], requires_grad=True)

    conv, div, corr = net.compute_PDE(X,Y,T)
    total = net.weights[0]*conv.abs() + net.weights[1]*div.abs() + net.weights[2]*corr.abs()
    total = total.data.cpu().reshape(Nx, Ny).transpose(1,0)
    
    plt.figure(dpi=200)
    plt.imshow(np.where(total-eps>=0, total, np.nan), cmap='magma', extent=[size[0],size[1],size[3],size[2]])
    plt.gca().invert_yaxis()
    plt.colorbar()
    # plt.clim(0, 1)
    
    mask = (net.t_PDE.data.cpu().numpy() == t)
    plt.scatter(net.x_PDE.data.cpu()[mask], net.y_PDE.data.cpu().numpy()[mask],alpha=0.5, c='lime', edgecolors='white', s=8, linewidths=0.5)
    plt.title(f't = {t} s')


def plot_boundary_fields(net, y, t, py, plot_linear:False):
    plt.figure(figsize=(15,12))
    
    Ny = y.shape[0]
    Nt = t.shape[0]
    y = torch.Tensor(y).to(net.device)
    t = net.T*torch.Tensor(t).to(net.device)

    mesh_yt = torch.stack(torch.meshgrid((y,t), indexing='ij')).reshape(2, -1)
    y, t = mesh_yt
    x_l = torch.zeros_like(y)
    x_r = torch.ones_like(y)
    c_l, p_l, _ = net.Boundary_conditions(x_l, y, t)
    _,   p_r, _ = net.Boundary_conditions(x_r, y, t)
    
    K = net.l(net.N_BC)**2
    N = net.N_BC2
    # TOP BOUNDARY
    plt.subplot(221)
    plt.title('Top')
    if plot_linear:
        plt.scatter(net.x_BC[:K].data.cpu(), net.t_BC[:K].data.cpu(),
                    alpha=0.5, c='lightblue', edgecolors='white', s=7, linewidths=0.5)
    plt.scatter(net.x_BC[4*K:4*K+N].data.cpu(), net.t_BC[4*K:4*K+N].data.cpu(),
              alpha=0.5, c='lime', edgecolors='white', s=7, linewidths=0.5)
    plt.imshow(np.abs(py[:,-1,:]), cmap='magma',extent=[0,1,1,0])
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('t')
    
    # BOTTOM BOUNDARY
    plt.subplot(222)
    plt.title('Bottom')
    if plot_linear:
        plt.scatter(net.x_BC[K:2*K].data.cpu(), net.t_BC[K:2*K].data.cpu(),
                    alpha=0.5, c='lightblue', edgecolors='white', s=7, linewidths=0.5)
    plt.scatter(net.x_BC[4*K+N:K+4*K+2*N].data.cpu(), net.t_BC[4*K+N:K+4*K+2*N].data.cpu(),
              alpha=0.5, c='lime', edgecolors='white', s=7, linewidths=0.5)
    plt.imshow(np.abs(py[:,0,:]), cmap='magma', extent=[0,1,1,0])
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('t')

    # LEFT BOUNDARY
    l = net.model([x_l, y, t])
    plt.subplot(223)
    plt.title('Left')
    if plot_linear:
        plt.scatter(net.y_BC[2*K:3*K].data.cpu(), net.t_BC[2*K:3*K].data.cpu(),
                    alpha=0.5, c='lightblue', edgecolors='white', s=7, linewidths=0.5)
    plt.scatter(net.y_BC[4*K+2*N:K+4*K+3*N].data.cpu(), net.t_BC[4*K+2*N:K+4*K+3*N].data.cpu(),
              alpha=0.5, c='lime', edgecolors='white', s=7, linewidths=0.5)
    plt.imshow(((l[:,0] - c_l).abs() + (l[:,1] - p_l).abs()).data.cpu().numpy().reshape(Ny,Nt).T,
               extent=[0,1,1,0], cmap='magma')
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.xlabel('y')
    plt.ylabel('t')
    
    # RIGHT BOUNDARY
    l = net.model([x_r, y, t])
    plt.subplot(224)
    plt.title('Right')
    if plot_linear:
        plt.scatter(net.y_BC[3*K:4*K].data.cpu(), net.t_BC[3*K:4*K].data.cpu(),
                    alpha=0.5, c='lightblue', edgecolors='white', s=7, linewidths=0.5)
    plt.scatter(net.y_BC[4*K+3*N:].data.cpu(), net.t_BC[4*K+3*N:].data.cpu(),
              alpha=0.5, c='lime', edgecolors='white', s=7, linewidths=0.5)
    plt.imshow((l[:,1] - p_r).abs().data.cpu().numpy().reshape(Ny,Nt).T,
               extent=[0,1,1,0], cmap='magma')
    plt.gca().invert_yaxis()
    plt.colorbar()  
    plt.xlabel('y')
    plt.ylabel('t')


def anim_result(data,
                steps,
                streamplot_data=False,
                clims=False,
                title=None,
                path="",
                name=None,
                colour='viridis',
                savetogif=False,
                showMe=False,
                limits=[0, 1, 0, 1],
                dpi=100,
                ):
    xmin, xmax, ymin, ymax = limits
    size_t = data.shape[0]
    size_y, size_x = data[0].shape

    path = Path(path or ".")
    name = name or f"t={size_t}, max_x={size_x}, max_y={size_y}"
    filename_base = path / name

    fig, ax = plt.subplots()
    fig.dpi = dpi

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel('x, m')
    ax.set_ylabel('y, m')

    title_str = title if title else 'f(x,y,t)'
    ttl = ax.set_title(f'{title_str} at t = 0')

    line = ax.imshow(data[0], cmap=colour, extent=limits, interpolation='none', origin='upper')
    plt.colorbar(line, ax=ax, label='Concentration $c, [M^0L^0T^0]$')

    if clims:
        line.set_clim(*clims)

    def animate(i):
        for coll in ax.collections[len(ax.collections) - 1::-1]:
            coll.remove()
        for patch in ax.patches:
            patch.remove()

        ttl.set_text(f'{title_str} at t = {np.round(i * steps, 3)}')

        if streamplot_data:
            ax.streamplot(
                streamplot_data[0],
                streamplot_data[1],
                streamplot_data[2][i],
                streamplot_data[3][i],
                linewidth=1,
                color='lightgrey',
                density=[1, 0.8]
            )

        line.set_array(data[i])
        return line, ttl

    ani = animation.FuncAnimation(fig, animate, interval=30, frames=size_t, blit=False)

    if showMe:
        plt.show()

    ext = ".gif" if savetogif else ".mp4"
    writer_cls = animation.PillowWriter if savetogif else animation.FFMpegWriter
    writer_args = {"fps": 30, "metadata": {"artist": "Doofenshmirtz Evil Inc."}}
    if not savetogif:
        writer_args["bitrate"] = 1800

    filename = filename_base.with_suffix(ext)
    counter = 1
    while filename.exists():
        filename = filename_base.with_name(f"{filename_base.stem}_{counter}").with_suffix(ext)
        counter += 1

    try:
        print(f"Saving animation to {filename} ...")
        writer = writer_cls(**writer_args)
        ani.save(str(filename), writer=writer)
        print("Done.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
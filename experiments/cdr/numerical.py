import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from tqdm import tqdm

import numpy as np

def tvd_advection(u, beta, dx, limiter='minmod', boundary='periodic'):
    N = len(u)
    u_ext = np.zeros(N + 4)
    u_ext[2:-2] = u
    if boundary == 'periodic':
        u_ext[:2] = u[-2:]
        u_ext[-2:] = u[:2]
    else:
        u_ext[:2] = u[0]
        u_ext[-2:] = u[-1]

    eps = 1e-8
    du_left  = u_ext[2:-2] - u_ext[1:-3]
    du_right = u_ext[3:-1] - u_ext[2:-2]
    r = du_left / (du_right + eps)

    if limiter == 'minmod':
        phi = np.maximum(0, np.minimum(1, r))
    elif limiter == 'vanleer':
        phi = (r + np.abs(r)) / (1 + np.abs(r) + eps)
    elif limiter == 'superbee':
        phi = np.maximum(0, np.maximum(np.minimum(1, 2*r), np.minimum(2, r)))
    else:
        raise ValueError("limiter must be 'minmod', 'vanleer' or 'superbee'")

    flux = np.zeros(N)
    for i in range(N):
        u_i   = u_ext[i+2]
        u_ip1 = u_ext[i+3]
        phi_i = phi[i]
        phi_ip1 = phi[(i+1) % N]

        if beta >= 0:
            u_face = u_i + 0.5 * phi_i * (u_ip1 - u_i)
        else:
            u_ip2 = u_ext[i+4]
            u_face = u_ip1 - 0.5 * phi_ip1 * (u_ip2 - u_ip1)
        flux[i] = beta * u_face

    du_dt_adv = np.zeros(N)
    for i in range(N):
        flux_left = flux[i-1]
        flux_right = flux[i]
        du_dt_adv[i] = -(flux_right - flux_left) / dx

    return du_dt_adv

def step(u_old, beta, nu, rho, dx, dt, f, limiter='minmod', boundary='periodic'):
    N = len(u_old)
    
    def L(u):
        adv = tvd_advection(u, beta, dx, limiter, boundary)
        
        if boundary == 'periodic':
            diff = nu * (np.roll(u, -1) - 2*u + np.roll(u, 1)) / dx**2
        else:
            diff = np.zeros_like(u)
            diff[1:-1] = nu * (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
        react = rho * f(u)
        
        return adv + diff + react
    
    k1 = L(u_old)
    k2 = L(u_old + 0.5*dt*k1)
    k3 = L(u_old + 0.5*dt*k2)
    k4 = L(u_old + dt*k3)
    
    u_new = u_old + dt/6.0 * (k1 + 2*k2 + 2*k3 + k4)
    
    return u_new

def solve_and_plot(beta, nu, rho, L, T, Nx, Nt, u0_func, f_func, limiter='vanleer'):
    dx = L / Nx
    dt = T / Nt
    x = np.linspace(0.5*dx, L - 0.5*dx, Nx)
    t = np.linspace(0, T, Nt)
    
    u = u0_func(x)
    U = np.zeros((Nt, Nx))
    U[0, :] = u.copy()
    
    for n in tqdm(range(0, Nt-1)):
        u = step(u, beta, nu, rho, dx, dt, f_func, limiter, boundary='periodic')
        U[n+1, :] = u.copy()
    
    fig = plt.figure(figsize=(10, 6))
    mesh = plt.pcolormesh(t, x, U.T, shading='auto', cmap='rainbow', norm=Normalize(vmin=U.min(), vmax=U.max()))
    plt.colorbar(mesh, label='u(x,t)')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title(rf'Решение $u(x,t)$: $\beta=${beta}, $\nu=${nu}, $\rho=${rho}')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    fig.savefig("images/cdr_numerical.png")
    
    return x, t, U

if __name__ == "__main__":
    beta = 2.0
    nu = 0.1
    rho = 4.0
    L = 2.0 * np.pi
    T = 1.0
    Nx = 200
    Nt = 400
    
    def u0(x):
        sigma = np.pi / 4
        return 0.1*np.exp(-(x - np.pi) ** 2 / (2.0 * sigma ** 2))

    def f_source(u):
        return u * (1 - u)
    
    x, t, U = solve_and_plot(beta, nu, rho, L, T, Nx, Nt, u0, f_source, limiter='vanleer')
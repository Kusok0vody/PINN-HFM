import math
import numpy as np


def _bessel():
    """scipy is an optional dependency — fail with a useful message, not an ImportError."""
    try:
        from scipy.special import jv, yv
    except ImportError as exc:
        raise ImportError(
            "Analytic references need scipy: pip install scipy"
        ) from exc
    return jv, yv


def helmholtz_annulus(
    x, y,
    k:             float,
    r_in:          float,
    r_out:         float,
    n_arcs:        int   = 8,
    n_terms:       int   = 64,
    resonance_tol: float = 1e-2,
):
    """
    Exact solution of the Helmholtz problem on an annulus.

        laplace(u) + k u = 0,        r_in < rho < r_out
        u = 0                        on rho = r_in
        u = sgn(sin(m theta))        on rho = r_out,   m = n_arcs / 2

    The outer boundary condition is the alternating +/-1 pattern over n_arcs
    equal arcs used by the experiment: arc i spans theta in [2 pi i / n_arcs,
    2 pi (i+1) / n_arcs) and carries +1 for even i.

    Separation of variables in polar coordinates gives a Bessel equation of
    order n in kappa*rho with kappa = sqrt(k), so each mode is a combination of
    J_n and Y_n. The combination vanishing at rho = r_in is

        C_n(rho) = Y_n(kappa r_in) J_n(kappa rho) - J_n(kappa r_in) Y_n(kappa rho)

    and the solution follows from the Fourier series of the square wave,
    sgn(sin(m theta)) = (4/pi) sum_{j odd} sin(j m theta) / j:

        u(rho, theta) = (4/pi) sum_{j odd} C_n(rho) / (j C_n(r_out)) sin(n theta),
                        n = m j

    Only orders n = m, 3m, 5m, ... contribute; all cosine modes vanish.

    Truncation. The number of retained terms depends only on the problem data
    (k, r_in, r_out, n_arcs, n_terms) and never on which points are queried.
    That matters: an evaluation-dependent cutoff would make this a different
    function for every grid it is asked about, and any finite-difference or
    error check against it would then measure the cutoff rather than the
    solution. The series stops early only when the Bessel functions at the two
    boundary radii cease to be representable in double precision, which is a
    property of the order n alone. Because |Y_n| decreases and |J_n| increases
    with the argument, finiteness at r_in and r_out implies finiteness
    throughout the annulus.

    Accuracy. Term n decays roughly like (rho / r_out)^n, so truncation is
    invisible in the interior but leaves Gibbs oscillations in a thin layer
    near the outer boundary, where the boundary datum is genuinely
    discontinuous. Expect ~1e-12 for rho < 0.9 r_out, and O(1/n_max) at the
    jumps — no truncation fixes that, the datum itself is not smooth.

    Args:
        x, y:          coordinate arrays of any matching shape (numpy or torch)
        k:             coefficient in laplace(u) + k u = 0; must be positive
        r_in:          inner radius
        r_out:         outer radius
        n_arcs:        number of alternating arcs on the outer boundary (even)
        n_terms:       maximum number of retained series terms
        resonance_tol: reject k when the modal determinant collapses to this
                       fraction of the terms forming it. The ratio maps
                       directly onto how far the solution is amplified above
                       its unit boundary datum — for the 1:3 annulus, 1.0 gives
                       max|u| ~ 6, 1e-1 gives ~3e2, 3e-3 gives ~1e4. The
                       default rejects amplification beyond roughly 1e3, where
                       the solution is dominated by a near-eigenmode and is
                       useless as a training target.

    Returns:
        numpy array of u, broadcast to the shape of (x, y). Points outside the
        annulus are NaN — the solution is not defined there, so masking is the
        caller's business and a silent extrapolation would be a trap.

    Raises:
        ValueError: on non-positive k, odd n_arcs, or a k at or near a
                    Dirichlet eigenvalue of the annulus, where the boundary
                    value problem has no unique solution
    """
    jv, yv = _bessel()

    if k <= 0.0:
        raise ValueError(
            f"helmholtz_annulus needs k > 0 (got {k}); k <= 0 is a different "
            "equation and needs modified Bessel functions."
        )
    if n_arcs % 2 != 0:
        raise ValueError(f"n_arcs must be even (got {n_arcs}).")
    if not 0.0 < r_in < r_out:
        raise ValueError(f"need 0 < r_in < r_out (got {r_in}, {r_out}).")

    x = np.asarray(getattr(x, "detach", lambda: x)().cpu() if hasattr(x, "detach") else x,
                   dtype=np.float64)
    y = np.asarray(getattr(y, "detach", lambda: y)().cpu() if hasattr(y, "detach") else y,
                   dtype=np.float64)

    shape = np.broadcast(x, y).shape
    rho   = np.hypot(x, y).ravel()
    theta = np.arctan2(y, x).ravel()

    # The solution exists only on the annulus. Evaluating the series outside it
    # is not merely wrong but singular: Y_n diverges as rho -> 0, so a cartesian
    # grid covering the inner hole would poison the sum with inf and NaN.
    #
    # The tolerance is sized for float32 callers, which is what the rest of the
    # project uses: a polar grid built as rho*cos(theta) in single precision
    # misses its own boundary radii by ~1e-7 relative, and a tighter tolerance
    # would silently turn whole boundary rings into NaN. Radii inside the
    # tolerance are clamped onto the boundary rather than extrapolated.
    eps    = 1e-6 * r_out
    inside = (rho >= r_in - eps) & (rho <= r_out + eps)
    rho    = np.clip(rho, r_in, r_out)

    kappa = math.sqrt(k)
    m     = n_arcs // 2

    z_in     = kappa * r_in
    z_out    = kappa * r_out
    z_rho    = kappa * rho[inside]
    theta_in = theta[inside]
    acc      = np.zeros_like(z_rho)

    used = 0
    for j in range(1, 2 * n_terms, 2):
        n = m * j

        j_in,  y_in  = jv(n, z_in),  yv(n, z_in)
        j_out, y_out = jv(n, z_out), yv(n, z_out)

        # Beyond some order J_n underflows to 0 and Y_n overflows to -inf.
        # This test involves n and the two boundary radii only, so the number
        # of retained terms is a property of the problem, not of the query.
        boundary = (j_in, y_in, j_out, y_out, y_in * j_out, j_in * y_out)
        if not all(np.isfinite(v) for v in boundary):
            break
        if j_out == 0.0 or y_in == 0.0:
            break

        denom = y_in * j_out - j_in * y_out
        scale = max(abs(y_in * j_out), abs(j_in * y_out))
        if abs(denom) <= resonance_tol * scale:
            raise ValueError(
                f"k = {k} is at or near a Dirichlet eigenvalue of this annulus "
                f"at mode n = {n}: the boundary value problem has no unique "
                f"solution there. Use helmholtz_annulus_resonances() to pick a "
                f"sweep that avoids these."
            )

        c_rho = y_in * jv(n, z_rho) - j_in * yv(n, z_rho)

        acc  += (4.0 / math.pi) * (c_rho / denom) * np.sin(n * theta_in) / j
        used += 1

    if used == 0:
        raise ValueError(
            f"no usable series terms for k = {k}, r_in = {r_in}, r_out = {r_out}; "
            "the Bessel functions underflow immediately."
        )

    u = np.full(rho.shape, np.nan)
    u[inside] = acc
    return u.reshape(shape)


def advection_diffusion_periodic(
    t, x,
    beta:     float,
    nu:       float,
    sigma:    float = math.pi / 4.0,
    x0:       float = math.pi,
    L:        float = 2.0 * math.pi,
    n_images: int   = 4,
):
    """
    Exact solution of the periodic advection-diffusion equation.

        du/dt + beta du/dx - nu d2u/dx2 = 0        on [0, L], periodic
        u(0, x) = exp(-(x - x0)^2 / (2 sigma^2))

    This is the CDR problem with the reaction term switched off (rho = 0),
    where a closed form exists: the Gaussian is advected at speed beta while
    its variance grows as sigma^2 + 2 nu t. Periodicity is imposed by summing
    over image sources displaced by multiples of L, which matters as soon as
    beta * t is comparable to L — a free-space Gaussian would place the bump
    outside the domain instead of wrapping it around.

    Note on the initial condition: CDR1D imposes the plain Gaussian, not its
    periodic extension. The two differ by the images at t = 0, about 3e-4 for
    the default sigma = pi/4, which bounds how exact this reference can be.
    Narrow bumps make that discrepancy negligible, wide ones do not.

    Args:
        t, x:     coordinate arrays of any matching shape (numpy or torch)
        beta:     advection speed
        nu:       diffusion coefficient; must be non-negative
        sigma:    initial Gaussian width
        x0:       initial centre
        L:        domain period
        n_images: how many image sources to sum on each side

    Returns:
        numpy array of u, broadcast to the shape of (t, x)
    """
    if nu < 0.0:
        raise ValueError(f"nu must be non-negative (got {nu}).")

    t = np.asarray(t.detach().cpu() if hasattr(t, "detach") else t, dtype=np.float64)
    x = np.asarray(x.detach().cpu() if hasattr(x, "detach") else x, dtype=np.float64)

    var = sigma ** 2 + 2.0 * nu * t
    amp = sigma / np.sqrt(var)

    u = np.zeros(np.broadcast(t, x).shape, dtype=np.float64)
    for m in range(-n_images, n_images + 1):
        shifted = x - x0 - beta * t + m * L
        u = u + amp * np.exp(-(shifted ** 2) / (2.0 * var))
    return u


def helmholtz_annulus_amplification(k, r_in, r_out, n_arcs=8, n_terms=16,
                                    n_radii=24):
    """
    How large the exact solution gets, for a boundary datum of size one.

    An upper bound on max|u|, from the same series as helmholtz_annulus but
    without its angular part: each mode contributes at most its radial profile
    times its Fourier coefficient, so

        A(k) = (4/pi) sum_j max_rho |C_n(rho)| / (j |C_n(r_out)|),   n = m j

    which is what the series would give if every mode peaked together. Cheap —
    a few dozen Bessel evaluations, no angular grid — so a sweep can test every
    drawn parameter before training on it.

    This replaces an earlier attempt that measured the modal determinant
    relative to the terms forming it. That number is not the amplification and
    reads almost perfectly innocent where the solution is already enormous: at
    k = 6.4 on the 1:3 annulus it gave 0.98 out of 1 while max|u| was 29.6, and
    it admitted k = 6.516 at 0.027 where max|u| is 986. The denominator of the
    series is the determinant itself, not its ratio to anything.

    Returns inf where the series cannot be formed at all, which is the same
    condition helmholtz_annulus raises on.
    """
    jv, yv = _bessel()
    if k <= 0.0:
        return float("inf")

    kappa = math.sqrt(k)
    m     = n_arcs // 2
    z_in, z_out = kappa * r_in, kappa * r_out
    z_rho = kappa * np.linspace(r_in, r_out, n_radii)

    total = 0.0
    for j in range(1, 2 * n_terms, 2):
        n = m * j
        j_in,  y_in  = jv(n, z_in),  yv(n, z_in)
        j_out, y_out = jv(n, z_out), yv(n, z_out)
        if not all(np.isfinite(v) for v in (j_in, y_in, j_out, y_out)):
            break
        denom = y_in * j_out - j_in * y_out
        if denom == 0.0:
            return float("inf")
        c_rho = y_in * jv(n, z_rho) - j_in * yv(n, z_rho)
        if not np.all(np.isfinite(c_rho)):
            break
        total += (4.0 / math.pi) * np.max(np.abs(c_rho)) / (abs(denom) * j)
    return float(total)


def helmholtz_annulus_resonances(k_min, k_max, r_in, r_out, n_arcs=8,
                                 n_modes=32, n_scan=20000):
    """
    Values of k in [k_min, k_max] where helmholtz_annulus is ill-posed.

    Useful before choosing a parameter sweep: at a Dirichlet eigenvalue of the
    annulus the boundary value problem has no unique solution, so neither the
    analytic reference nor the PINN has a well-defined target there.

    Returns:
        sorted list of approximate resonant k values
    """
    jv, yv = _bessel()

    m  = n_arcs // 2
    ks = np.linspace(max(k_min, 1e-9), k_max, n_scan)
    kappas = np.sqrt(ks)

    found = []
    for j in range(1, 2 * n_modes, 2):
        n = m * j
        with np.errstate(over="ignore", invalid="ignore"):
            d = (yv(n, kappas * r_in) * jv(n, kappas * r_out)
                 - jv(n, kappas * r_in) * yv(n, kappas * r_out))
        good = np.isfinite(d)
        if good.sum() < 2:
            continue
        d  = d[good]
        kk = ks[good]
        sign_change = np.nonzero(np.sign(d[:-1]) * np.sign(d[1:]) < 0)[0]
        found.extend(0.5 * (kk[sign_change] + kk[sign_change + 1]))

    return sorted(found)


def lotka_volterra(t, y0, par, dt_max=1e-2):
    """
    Three-species Lotka-Volterra reference, integrated rather than solved.

    dG/dt = r G (1 - G/K) - alpha G H
    dH/dt = e1 alpha G H - d1 H - beta H P
    dP/dt = e2 beta H P - d2 P

    There is no closed form, so this is classical RK4 on a fixed grid, dense
    enough that the discretisation error sits far below anything a network is
    going to be judged against, and the result is then read off at the
    requested times by linear interpolation.

    Fixed step rather than adaptive on purpose. The system is not stiff over
    the ranges this sweep uses, an adaptive controller would need scipy, which
    this project does not depend on, and a fixed grid makes the cost the same
    for every parameter setting — which matters when the caller is a training
    loop redrawing its batch every hundred steps.

    Every setting is integrated at once: the state is (M, 3) and the whole
    sweep advances in one loop, so the cost is one integration, not M of them.

    Args:
        t:      (N,) times to report, any order, all >= 0
        y0:     (3,) or (M, 3) initial values of G, H, P
        par:    dict of name -> (M,) arrays, keys r K alpha e1 d1 beta e2 d2
        dt_max: upper bound on the integration step. Also bounded internally
                by the fastest rate in par, so the default stays safe across a
                sweep instead of only at the parameters it was tuned on.

    Returns:
        (N, M, 3) array, last axis ordered G, H, P
    """
    import numpy as np

    keys = ("r", "K", "alpha", "e1", "d1", "beta", "e2", "d2")
    p = {k: np.asarray(par[k], dtype=np.float64).reshape(-1) for k in keys}
    m = max(v.size for v in p.values())
    p = {k: np.broadcast_to(v, (m,)).copy() for k, v in p.items()}

    t = np.asarray(t, dtype=np.float64).reshape(-1)
    if t.min() < 0:
        raise ValueError("lotka_volterra reports from t = 0 forward; got "
                         f"t_min = {t.min()}")

    y0 = np.asarray(y0, dtype=np.float64)
    y  = np.broadcast_to(y0.reshape(-1, 3) if y0.ndim > 1 else y0.reshape(1, 3),
                         (m, 3)).copy()

    def rhs(s):
        G, H, P = s[:, 0], s[:, 1], s[:, 2]
        return np.stack([
            p["r"] * G * (1.0 - G / p["K"]) - p["alpha"] * G * H,
            p["e1"] * p["alpha"] * G * H - p["d1"] * H - p["beta"] * H * P,
            p["e2"] * p["beta"] * H * P - p["d2"] * P,
        ], axis=1)

    t_end = float(t.max())
    if t_end == 0.0:
        return np.broadcast_to(y, (t.size, m, 3)).copy()

    # Bound the step by the fastest process in the batch, not only by the
    # caller's number. A default chosen at one set of parameters says nothing
    # about a sweep that multiplies every rate by three, and the failure would
    # be silent: an integration that is merely less accurate, reported with the
    # same confidence. Rates are read off the equations — growth, grazing,
    # conversion, predation, death — with the populations bounded by K.
    scale = np.maximum(p["K"], np.abs(y).max(axis=1) if y.size else 1.0)
    rate  = np.max(np.stack([
        p["r"], p["d1"], p["d2"],
        p["alpha"] * scale, p["e1"] * p["alpha"] * scale,
        p["beta"] * scale, p["e2"] * p["beta"] * scale,
    ]))
    # A twentieth of the fastest timescale: comfortably inside RK4's stability
    # region and well past the point where its truncation error matters.
    dt_max = min(float(dt_max), 0.05 / max(float(rate), 1e-12))

    n_steps = max(1, int(np.ceil(t_end / dt_max)))
    dt      = t_end / n_steps
    grid    = np.linspace(0.0, t_end, n_steps + 1)
    traj    = np.empty((n_steps + 1, m, 3))
    slope   = np.empty((n_steps + 1, m, 3))
    traj[0] = y

    for i in range(n_steps):
        k1 = rhs(y)
        k2 = rhs(y + 0.5 * dt * k1)
        k3 = rhs(y + 0.5 * dt * k2)
        k4 = rhs(y + dt * k3)
        slope[i] = k1
        y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        # The populations are non-negative quantities and the equations keep
        # them so; only rounding can push one a hair below zero, and left alone
        # a negative G feeds the logistic term and diverges. Clipping is a
        # guard against arithmetic, not a correction to the model.
        np.maximum(y, 0.0, out=y)
        traj[i + 1] = y
    slope[n_steps] = rhs(y)

    # Cubic Hermite between grid points, not linear.
    #
    # The reported times are wherever the caller's collocation points fell, so
    # they almost never land on the grid and the interpolation is what the
    # caller actually receives. Linear would make it the dominant error at
    # O(dt^2 |y''|) — coarser than the RK4 step that produced the values, which
    # would be paying for a fourth-order integrator and reading it second
    # order. The exact slope at every grid point is already in hand, since it
    # is the right-hand side and k1 is precisely that, so the cubic through
    # both values and both slopes costs one array of stored k1 and brings the
    # interpolation to O(dt^4), back in line with the integration.
    idx = np.clip(np.searchsorted(grid, t, side="right") - 1, 0, n_steps - 1)
    u   = ((t - grid[idx]) / dt)[:, None, None]
    u2, u3 = u * u, u * u * u
    return (
        (2.0 * u3 - 3.0 * u2 + 1.0) * traj[idx]
        + (u3 - 2.0 * u2 + u) * dt * slope[idx]
        + (-2.0 * u3 + 3.0 * u2) * traj[idx + 1]
        + (u3 - u2) * dt * slope[idx + 1]
    )

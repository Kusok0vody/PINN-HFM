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

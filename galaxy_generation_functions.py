import numpy as np
from enum import StrEnum
import matplotlib.pyplot as plt
from typing import Literal, overload

class Type(StrEnum):
    Star = "1"
    Dark = "2"

def virial_ellipsoid(pos, mass, a, b, c, G = 1.0, seed=None,
    disp: float = 0.1,             # fraction of v_c used as in-plane dispersion
    spin_axis: tuple[float, float, float] | None = None,  # preferred global spin axis; None -> z
):
    """
    Initialize velocities for an ellipsoidal configuration with tangential support.

    - Uses ellipsoidal shells defined by (x/a)^2 + (y/b)^2 + (z/c)^2 = const.
    - Mean speed set to v_c(r_eff) = sqrt(G * M_enc / r_eff) where r_eff ≈ m * (abc)^(1/3).
    - Velocities lie in the tangent plane (no radial component). Optional in-plane dispersion.
    - Finally rescales K so that sum m |v|^2 ≈ sum m v_c^2, similar to the original code.

    This avoids strong initial in-fall by eliminating radial motion at t = 0.
    """

    rng = np.random.default_rng(seed)
    eps = 1e-12 # for avoiding division by 0

    # Center positions at centre of mass
    x = pos - np.average(pos, weights=mass, axis=0)

    # Ellipsoidal radius m and volume-equivalent spherical radius r_eff
    inv2 = np.array([1.0 / (a * a), 1.0 / (b * b), 1.0 / (c * c)], dtype=float)
    m = np.sqrt(np.maximum((x * x * inv2).sum(axis=1), eps))
    r = np.maximum(m * (abs(a * b * c)) ** (1.0/3.0), eps)  # r_eff

    # Enclosed mass M_enc(r): cumulative sum over particles sorted by r
    order = np.argsort(r)
    Menc = np.empty_like(r)
    Menc[order] = np.cumsum(mass[order])

    # Circular speed (spherical approximation on r_eff)
    v_c = np.sqrt(G * Menc / r)

    # Ellipsoidal outward normal (gradient of m)
    n = x * inv2  # proportional to grad(m)
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + eps)  # unit "radial" vector on the shell

    # Choose a global spin axis (default: z / c-axis). We'll build a tangent basis from it.
    if spin_axis is None:
        k = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        k = np.array(spin_axis, dtype=float)
    k /= (np.linalg.norm(k) + eps)
    k = np.broadcast_to(k, x.shape)

    # First tangent direction: perpendicular to both n and k
    t1 = np.cross(k, n)
    t1_norm = np.linalg.norm(t1, axis=1, keepdims=True)

    # Fallback where k ~ n (degenerate cross): use an alternate axis
    bad = (t1_norm[:, 0] < 1e-8)
    if np.any(bad):
        alt = np.array([0.0, 1.0, 0.0], dtype=float)
        t1_alt = np.cross(np.broadcast_to(alt, x[bad].shape), n[bad])
        t1_alt /= (np.linalg.norm(t1_alt, axis=1, keepdims=True) + eps)
        t1[bad] = t1_alt
        t1_norm = np.linalg.norm(t1, axis=1, keepdims=True)

    t1 /= (t1_norm + eps)

    # Second tangent direction in the shell
    t2 = np.cross(n, t1)  # already unit-length if n,t1 are orthonormal

    # Mean tangential motion + in-plane (t1,t2) Gaussian dispersion
    z1 = rng.standard_normal(len(x))
    z2 = rng.standard_normal(len(x))
    v = (
        t1 * v_c[:, None]  # circular support
        + (t1 * z1[:, None] + t2 * z2[:, None]) * (disp * v_c)[:, None]  # in-plane dispersion
    )

    # Project out any tiny radial component from numerical error
    v -= n * (v * n).sum(axis=1, keepdims=True)

    # Remove bulk drift
    v -= np.average(v, weights=mass, axis=0)

    # Rescale kinetic energy so that sum m |v|^2 ≈ sum m v_c^2 (same intent as original)
    num = np.sum(mass * v_c * v_c)
    den = np.sum(mass * (v * v).sum(axis=1))
    if den > eps:
        v *= np.sqrt(num / den)

    return v

#Equal mass disk sampling
def generate_disk_equal_mass(N_particles: int, Rd: float, z0: float, M_tot: float = 1.0, Rmax: float = None, zmax: float = None, seed: int = None):
    """
    Equal-mass sampling directly from an exponential disk:
      ρ(R, z) ∝ exp(-R/Rd) * exp(-|z|/z0)
    - R is Gamma(k=2, θ=Rd) distributed (surface density ∝ exp(-R/Rd)).
    - z is Laplace(0, z0) (vertical exponential).
    Optional truncation with Rmax, zmax via rejection.
    Returns positions (N,3), velocities (N,3 zeros), masses (N)
    """
    rng = np.random.default_rng(seed)
    if Rmax is None: Rmax = 5.0 * Rd
    if zmax is None: zmax = 5.0 * z0

    R = []
    # Rejection for truncated R
    while len(R) < N_particles:
        # Draw in chunks of candidate R from Gamma(k=2, θ=Rd)
        chunk = rng.gamma(shape=2.0, scale=Rd, size=max(4*(N_particles-len(R)), 1024))
        chunk = chunk[chunk <= Rmax]
        R.extend(chunk.tolist())
    R = np.array(R[:N_particles])

    phi = rng.uniform(0.0, 2.0*np.pi, size=N_particles)

    # Truncated Laplace for z via rejection
    z = []
    while len(z) < N_particles:
        chunk = rng.laplace(loc=0.0, scale=z0, size=max(4*(N_particles-len(z)), 1024))
        chunk = chunk[np.abs(chunk) <= zmax]
        z.extend(chunk.tolist())
    z = np.array(z[:N_particles])

    x = R * np.cos(phi)
    y = R * np.sin(phi)
    pos = np.column_stack([x, y, z])

    vel = np.zeros_like(pos)
    mass = np.full(N_particles, M_tot / N_particles, dtype=float)

    return pos, vel, mass

def generate_disk_importance_mass(N_particles: int, Rd: float, z0: float, M_tot: float = 1.0, Rmax: float = None, zmax: float = None, max_mass_ratio: float = None, seed: int = None):
    """
    Importance sampling with variable particle masses:
      - Sample particles approximately uniformly in a bounding cylinder.
      - Assign particle masses m_i ∝ ρ(R_i, z_i) so denser regions get larger masses.
      - Normalize Σ m_i = M_tot.
    This reduces particle clustering in the centre while preserving the target density in expectation.

    Optional:
      max_mass_ratio: if set (e.g. 10), clamp masses around the mean to limit two-body noise, then renormalise.

    Returns positions (N,3), velocities (N,3 zeros), masses (N)
    """
    rng = np.random.default_rng(seed)
    if Rmax is None: Rmax = 5.0 * Rd
    if zmax is None: zmax = 5.0 * z0

    # Uniform-in-volume cylinder sampling:
    # R pdf ∝ R on [0, Rmax] => R = Rmax * sqrt(u)
    u = rng.uniform(0.0, 1.0, size=N_particles)
    R = Rmax * np.sqrt(u)
    phi = rng.uniform(0.0, 2.0*np.pi, size=N_particles)
    z = rng.uniform(-zmax, zmax, size=N_particles)

    x = R * np.cos(phi)
    y = R * np.sin(phi)
    pos = np.column_stack([x, y, z])

    # Weights from target density shape
    w = np.exp(-R / Rd) * np.exp(-np.abs(z) / z0)
    # Normalise to total mass
    mass = M_tot * (w / np.sum(w))

    if max_mass_ratio is not None and max_mass_ratio > 1:
        m_mean = M_tot / N_particles
        m_min = m_mean / max_mass_ratio
        m_max = m_mean * max_mass_ratio
        mass = np.clip(mass, m_min, m_max)
        mass *= (M_tot / np.sum(mass))  # renormalize

    vel = np.zeros_like(pos)
    return pos, vel, mass

def generate_ellipse_importance(
    N_particles: int,
    a: float,
    b: float,
    c: float,
    M_tot: float = 1.0,
    seed: int | None = None,
):
    """
    Ellipsoidal galaxy generator using importance sampling.
    Target density (unnormalized): rho(m) ∝ exp(-m),
    with m^2 = (x/a)^2 + (y/b)^2 + (z/c)^2.

    Proposal: Anisotropic Gaussian N(0, diag([a^2, b^2, c^2])).
    Weights: w ∝ p(x) / q_unnorm(x); normalized so masses sum to M_tot.

    Returns a dict with:
      - "pos": (N, 3) positions
      - "mass": (N,) masses summing to M_tot
    """
    rng = np.random.default_rng(seed)

    # Sample from the proposal aligned with the ellipsoid axes
    sigmas = np.array([a, b, c], dtype=np.float64)
    pos = rng.normal(0.0, sigmas, size=(N_particles, 3)).astype(np.float64, copy=False)

    # Ellipsoidal radius m and m^2
    inv_axes_sq = np.array([1.0/(a*a), 1.0/(b*b), 1.0/(c*c)], dtype=np.float64)
    m2 = np.einsum("ij,j,ij->i", pos, inv_axes_sq, pos)
    m = np.sqrt(m2)

    # Target and proposal (unnormalized) log-densities
    log_p = -m                    # log p ∝ -m (gradual)
    log_q_unnorm = -0.5 * m2      # since q ∝ exp(-0.5 * sum((x_i/sigma_i)^2)) and sigmas=[a,b,c]

    # Importance weights (stable)
    log_w = log_q_unnorm - log_p
    log_w -= np.max(log_w)
    w = np.exp(log_w)
    w /= np.sum(w)

    mass = M_tot * w

    vel = virial_ellipsoid(pos, mass, a=a, b=b, c=c, G=1.0)
    return pos, vel, mass

def generate_diffuse_sphere(N_particles: int, R: float = 10.0, M_tot: float = 1.0, seed: int | None = None):
    rng = np.random.default_rng(seed)
    masses = np.full(N_particles, M_tot / N_particles, dtype=float)
    eps = 1e-12

    # Positions: exponential radius with isotropic directions
    r = rng.exponential(scale=R, size=N_particles)
    dirs = rng.normal(size=(N_particles, 3))
    dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + eps)
    positions = dirs * r[:, None]

    # Velocities: simple isotropic Maxwellian (rough virial-scale) G=1
    sigma_v = 0.6 * np.sqrt(M_tot / (R + eps))
    velocities = rng.normal(scale=sigma_v, size=(N_particles, 3))

    # Remove tiny bulk drifts
    positions -= np.average(positions, weights=masses, axis=0)
    velocities -= np.average(velocities, weights=masses, axis=0)

    return positions, velocities, masses

def generate_diffuse_sphere_importance(N_particles, R, M_tot, seed=None):
    """
     Inputs:
       - N_particles: number of particles
       - R: Plummer scale length 'a'
       - M_tot: total mass
       - seed: RNG seed

     Outputs:
       - positions: (N, 3) ndarray
       - velocities: (N, 3) ndarray (purely tangential)
       - masses: (N,) ndarray

     Notes:
       - Positions are drawn from a 3D normal proposal q(x) ~ N(0, a^2 I).
       - Masses use importance sampling to approximate a Plummer density:
           ρ_plummer(r) ∝ (1 + (r/a)^2)^(-5/2)
           w_i ∝ ρ_plummer(r_i) / q(x_i)
         Masses are normalised to sum to M_tot.
       - Velocities are set to circular speeds v = sqrt(M_enc(r)/r) with tangential directions.
         M_enc(r) is computed from the cumulative sum of the discrete masses.
     """
    rng = np.random.default_rng(seed)

    # Proportions of particle populations by mass
    Stellar_fraction = 0.7
    M_stellar = Stellar_fraction * M_tot
    M_dark = M_tot - M_stellar

    # Labels by row order
    N_star = int(round(Stellar_fraction * N_particles))
    N_dark = N_particles - N_star
    # Use dtype that fits 'Stellar'
    labels = np.empty(N_particles, dtype='U5')
    labels[:N_star] = Type.Star.value
    labels[N_star:] = Type.Dark.value

    print(labels)

    # 1) Positions: 3D normal proposal q(x) with std=R (simple and isotropic)
    positions = rng.normal(scale=R, size=(N_particles, 3)).astype(float)
    r = np.linalg.norm(positions, axis=1)
    x = r / R

    # 2) Importance weights for Plummer density vs. 3D normal proposal
    #    Use log-weights for numerical stability; constants cancel after normalisation.
    #    ρ*(r) ∝ (1 + x^2)^(-5/2), q*(x) ∝ exp(-x^2/2)  =>  w ∝ (1 + x^2)^(-5/2) * exp(+x^2/2)
    logw = -2.5 * np.log1p(x * x) + 0.5 * (x * x)
    logw -= np.max(logw)
    w = np.exp(logw)
    masses = (M_tot * w / np.sum(w)).astype(float)

    # 3) Tangential circular velocities from enclosed mass of the discrete system
    #    - Sort by radius, build cumulative enclosed mass
    idx = np.argsort(r)
    r_sorted = r[idx]
    m_sorted = masses[idx]
    Menc_sorted = np.cumsum(m_sorted)  # includes self; simple and stable

    #    - Circular speed v = sqrt(M_enc(r)/r); handle r=0 safely
    v_sorted = np.zeros(N_particles, dtype=float)
    nonzero = r_sorted > 0
    v_sorted[nonzero] = np.sqrt(Menc_sorted[nonzero] / r_sorted[nonzero])

    #    - Map v back to original particle order
    vmag = np.empty(N_particles, dtype=float)
    vmag[idx] = v_sorted

    # 4) Build tangential unit vectors perpendicular to r-hat
    rhat = np.zeros_like(positions)
    safe = r > 0
    rhat[safe] = positions[safe] / r[safe, None]

    q = rng.normal(size=(N_particles, 3))
    tang = np.cross(rhat, q)
    tn = np.linalg.norm(tang, axis=1)

    # Fallback for near-parallel cases
    fix = tn < 1e-12
    if np.any(fix):
        q[fix] = np.array([1.0, 0.0, 0.0])
        tang[fix] = np.cross(rhat[fix], q[fix])
        tn[fix] = np.linalg.norm(tang[fix], axis=1)

    # For r=0, pick any random unit direction; speed will be zero anyway
    if np.any(~safe):
        rnd = rng.normal(size=((~safe).sum(), 3))
        rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
        tang[~safe] = rnd
        tn[~safe] = 1.0

    tang /= tn[:, None]

    # 5) Final velocities: tangential direction with circular speed
    velocities = tang * vmag[:, None]

    return positions, velocities, masses, labels


def save_galaxy_npz(path, positions, masses, velocities=None):
    """
    Save a configuration you can reload later.
    path: e.g., "disk_config.npz"
    Arrays are stored as float64; meta is stored as a JSON string.
    """
    if velocities is None:
        velocities = np.zeros_like(positions)
    np.savez(file=path, pos=positions, vel=velocities, mass=masses)

def view_configuration(positions, masses, labels, title=None):
    xy = positions[:, :2]
    stellar = xy[labels == Type.Star.value]
    dark = xy[labels == Type.Dark.value]
    m_star = masses[labels == Type.Star.value]
    m_dark = masses[labels == Type.Dark.value]
    s_star = (m_star/np.min(m_star))  # Ratio scaling (Scaling error for equal masses)
    s_dark = (m_dark/np.min(m_dark))
    plt.style.use('dark_background')
    plt.figure(figsize=(5, 5))
    plt.scatter(stellar[:, 0], stellar[:, 1], s=s_star, c='yellow', alpha=0.5, linewidths=0)
    plt.scatter(dark[:, 0], dark[:, 1], s=s_dark, c='purple', alpha=0.5, linewidths=0)
    ax = plt.gca()
    ax.set_aspect('equal', 'box')
    xr = xy[:, 0].max() - xy[:, 0].min()
    yr = xy[:, 1].max() - xy[:, 1].min()
    pad = 1e-9 + 0.05 * max(xr, yr)
    ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
    ax.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    if title: ax.set_title(title)
    plt.savefig(f"Outputs/{title}.png")
    plt.tight_layout(); plt.show()

#Disk Overload
@overload
def generate_galaxy(
    morphology: Literal["disk"],
    sampling: Literal["equal", "importance"] = "importance",
    N_particles: int = 10000,
    Rd: float = None,
    z0: float = None,
    M_tot: float = 1.0,
    seed: int = None,
) -> np.ndarray:
    pass

#Ellipse Overload
@overload
def generate_galaxy(
    morphology: Literal["ellipse"],
    sampling: Literal["importance"] = "importance",
    N_particles: int = 10000,
    a: float = 1.0,
    b: float = 1.0,
    c: float = 1.0,
    M_tot: float = 1.0,
    seed: int = None
) -> np.ndarray:
    pass

# Diffuse Sphere Overload
@overload
def generate_galaxy(
        morphology: Literal["diffuse_sphere"],
        sampling: Literal["importance"] = "importance",
        N_particles: int = 10000,
        R: float = 10.0,
        M_tot: float = 1.0,
        seed: int = None
)  -> np.ndarray:
    pass

def generate_galaxy(
        morphology: Literal["disk", "ellipse", "diffuse_sphere"],
        sampling: Literal["equal", "importance"] = "importance",
        N_particles: int = 10000,
        Rd: float | None = None,
        R: float | None = None,
        z0: float | None = None,
        a: float | None = None,
        b: float | None = None,
        c: float | None = None,
        M_tot: float | None = 1.0,
        dm_factor: float = 0.0,
        seed: int | None = None,
):
    if morphology == "disk":
        if Rd is None: Rd = 10.0
        if z0 is None: z0 = 0.1
        if sampling == "equal":
            positions, velocities, masses = generate_disk_equal_mass(N_particles, Rd, z0, M_tot, seed)
        elif sampling == "importance":
            positions, velocities, masses = generate_disk_importance_mass(N_particles, Rd, z0, M_tot, seed)
        else:
            raise ValueError(f"Unknown sampling: {sampling!r}")
    elif morphology == "ellipse":
            positions, velocities, masses = generate_ellipse_importance(N_particles, a, b, c, M_tot, seed)
    elif morphology == "diffuse_sphere":
            positions, velocities, masses, labels = generate_diffuse_sphere_importance(N_particles, R, M_tot, seed)
    else:
        raise ValueError(f"Unknown morphology: {morphology!r}")
    save_galaxy_npz(f"Outputs/{morphology}_{sampling}_{seed}.npz", positions, masses, velocities)
    view_configuration(positions, masses, labels, title=f"{morphology}_{sampling}_{seed}")

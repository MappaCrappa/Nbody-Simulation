import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, overload

# Random generator
def _default_rng(seed=None):
    return np.random.default_rng(seed)

#Unnormalised 3D disk density shape: exp(-R/Rd) * exp(-|z|/z0)
def _exp_disk_weight(R, z, Rd, z0):
    return np.exp(-R / Rd) * np.exp(-np.abs(z) / z0)

#Equal mass disk sampling
def generate_disk_equal_mass(
    N_particles: int,
    Rd: float,
    z0: float,
    M_tot: float = 1.0,
    Rmax: float = None,
    zmax: float = None,
    seed: int = None
):
    """
    Equal-mass sampling directly from an exponential disk:
      ρ(R, z) ∝ exp(-R/Rd) * exp(-|z|/z0)
    - R is Gamma(k=2, θ=Rd) distributed (surface density ∝ exp(-R/Rd)).
    - z is Laplace(0, z0) (vertical exponential).
    Optional truncation with Rmax, zmax via rejection.
    Returns positions (N,3), velocities (N,3 zeros), masses (N)
    """
    rng = _default_rng(seed)
    if Rmax is None: Rmax = 5.0 * Rd
    if zmax is None: zmax = 5.0 * z0
    #if M_tot is None: M_tot = N_particles

    R = []
    # Rejection for truncated R
    while len(R) < N_particles:
        # Draw in chunks
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

def generate_disk_importance_mass(N_particles, Rd, z0, M_tot: float = 1.0, Rmax=None, zmax=None, max_mass_ratio=None, seed=None):
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
    rng = _default_rng(seed)
    if Rmax is None: Rmax = 5.0 * Rd
    if zmax is None: zmax = 5.0 * z0
    #if M_tot is None: M_tot = 1

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
    w = _exp_disk_weight(R, z, Rd, z0)
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

def save_galaxy_npz(path, positions, masses, velocities=None):
    """
    Save a configuration you can reload later.
    path: e.g., "disk_config.npz"
    Arrays are stored as float64; meta is stored as a JSON string.
    """
    if velocities is None:
        velocities = np.zeros_like(positions)
    np.savez(file=path, pos=positions, vel=velocities, mass=masses)

def view_configuration(positions, masses=None, title=None):
    xy = positions[:, :2]
    if masses is not None:
        s = 5.0 * (masses / (masses.mean() + 1e-12))
        s = np.maximum(2.0, np.minimum(20.0, s))
    else:
        s = 4.0
    plt.figure(figsize=(5, 5))
    plt.scatter(xy[:, 0], xy[:, 1], s=s, c='k', alpha=0.5, linewidths=0)
    ax = plt.gca()
    ax.set_aspect('equal', 'box')
    xr = xy[:, 0].max() - xy[:, 0].min()
    yr = xy[:, 1].max() - xy[:, 1].min()
    pad = 1e-9 + 0.05 * max(xr, yr)
    ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
    ax.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    if title: ax.set_title(title)
    plt.tight_layout(); plt.show()

@overload
def generate_galaxy(
    morphology: Literal["disk"],
    sampling: Literal["equal", "importance"],
    N_particles: int,
    Rd: float,
    z0: float,
    M_tot: float = 1.0,
    seed: int = None,
) -> generate_disk_equal_mass or generate_disk_importance_mass:
    pass

def generate_galaxy(
        morphology: Literal["disk"],
        sampling: Literal["equal", "importance"],
        N_particles: int,
        Rd: float | None = None,
        z0: float | None = None,
        M_tot: float = 1.0,
        seed: int | None = None,
):
    if morphology == "disk":
        if sampling == "equal":
            positions, velocities, masses= generate_disk_equal_mass(N_particles, Rd, z0, M_tot, seed)
        elif sampling == "importance":
            positions, velocities, masses= generate_disk_importance_mass(N_particles, Rd, z0, M_tot, seed)
        else:
            raise ValueError(f"Unknown sampling: {sampling!r}")
    else:
        raise ValueError(f"Unknown morphology: {morphology!r}")
    save_galaxy_npz(f"Outputs/{morphology}_{sampling}_{seed}.npz", positions, velocities, masses)
    view_configuration(positions, masses, title=sampling+"-mass sampling")

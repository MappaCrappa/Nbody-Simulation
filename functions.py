import numpy as np
from scipy.fft import fftn, ifftn

# Nearest Grid Point assignment
def NGP(positions, grid_size, box_size, masses):
    density = np.zeros((grid_size,)*3, dtype=float)
    positions = positions % box_size                                                                        # Wrapped positions across box edges
    indices = np.floor(positions / box_size * grid_size).astype(int)                                        # Normalize to grid indices
    np.add.at(density, (indices[:, 0], indices[:, 1], indices[:, 2]), masses)                        # Mass Counting
    return density

# Cloud-in-Cell assignment
def CIC(positions, grid_size, box_size, masses):
    density = np.zeros((grid_size,)*3, dtype=float)
    positions = positions % box_size
    cell_size = box_size / grid_size
    # Find normalized cell indices and weights
    scaled_pos = positions / cell_size
    i = np.floor(scaled_pos).astype(int)
    d = scaled_pos - i
    for p in range(positions.shape[0]):
        base = i[p]
        w = d[p]
        for dx in [0, 1]:
            wx = (1 - w[0]) if dx == 0 else w[0]
            x = (base[0] + dx) % grid_size
            for dy in [0, 1]:
                wy = (1 - w[1]) if dy == 0 else w[1]
                y = (base[1] + dy) % grid_size
                for dz in [0, 1]:
                    wz = (1 - w[2]) if dz == 0 else w[2]
                    z = (base[2] + dz) % grid_size
                    weight = wx * wy * wz
                    density[x, y, z] += masses[p] * weight
    return density

# Potential
def compute_potential(density, grid_size):
    k = np.fft.fftfreq(grid_size)
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    k2 = kx**2 + ky**2 + kz**2
    k2[0, 0, 0] = 1     #Division by 0 clause
    #density_k = fftn(density)
    density_mean = np.mean(density)
    density_k = fftn(density - density_mean)
    potential_k = density_k / k2
    potential_k[0, 0, 0] = 0.0  # Set mean of potential to zero (removes constant offset)
    potential = np.real(ifftn(potential_k))
    return potential

#Force
def force_NGP(potential, positions, grid_size, box_size):
    grad_x = np.gradient(potential, axis=0)
    grad_y = np.gradient(potential, axis=1)
    grad_z = np.gradient(potential, axis=2)
    indices = (positions / box_size * grid_size).astype(int) % grid_size
    ix, iy, iz = indices[:, 0], indices[:, 1], indices[:, 2]
    # Interpolate force from gradients at particle grid locations
    forces = np.stack([grad_x[ix, iy, iz], grad_y[ix, iy, iz], grad_z[ix, iy, iz]], axis=1)
    return forces

def force_CIC(potential, positions, grid_size, box_size):
    grad_x = np.gradient(potential, axis=0)
    grad_y = np.gradient(potential, axis=1)
    grad_z = np.gradient(potential, axis=2)
    # Combine into force_grid for easier access
    force_grid = np.stack([grad_x, grad_y, grad_z], axis=-1)  # shape (Nx,Ny,Nz,3)
    N_particles = positions.shape[0]
    cell_size = box_size / grid_size
    forces = np.zeros((N_particles, 3))
    positions = positions % box_size

    scaled_pos = positions / cell_size
    i = np.floor(scaled_pos).astype(int)
    d = scaled_pos - i

    for p in range(N_particles):
        base = i[p]
        w = d[p]
        for dx in [0, 1]:
            wx = (1 - w[0]) if dx == 0 else w[0]
            x = (base[0] + dx) % grid_size
            for dy in [0, 1]:
                wy = (1 - w[1]) if dy == 0 else w[1]
                y = (base[1] + dy) % grid_size
                for dz in [0, 1]:
                    wz = (1 - w[2]) if dz == 0 else w[2]
                    z = (base[2] + dz) % grid_size
                    weight = wx * wy * wz
                    forces[p] += force_grid[x, y, z] * weight
    return forces


#Diagnostic Functions
def compute_kinetic_energy(velocities, masses):
    KE = 0.5 * np.sum(masses * np.sum(velocities**2, axis=1))
    return KE

def compute_momentum(velocities, masses):
    return np.sum(masses[:, np.newaxis] * velocities, axis=0)

def compute_potential_energy(positions, masses, potential, grid_size, box_size):
    # Assign particle to nearest grid for potential
    indices = (positions / box_size * grid_size).astype(int) % grid_size
    phi_vals = potential[indices[:,0], indices[:,1], indices[:,2]]
    PE = np.sum(masses * phi_vals) * 0.5  # Factor 0.5 to avoid double-counting
    return PE
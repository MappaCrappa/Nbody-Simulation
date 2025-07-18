import numpy as np
from scipy.fft import fftn, ifftn

#Nearest Grid Point assignment
def NGP(positions, grid_size, box_size, masses):
    density = np.zeros((grid_size, grid_size, grid_size), dtype=float)
    #Wrapped positions across box edges
    positions = positions % box_size
    # Normalize to grid indices
    indices = np.floor(positions / box_size * grid_size).astype(int)
    # Only use in-bounds indices
    #indices_in = indices[(indices[:, 0] >= 0) & (indices[:, 0] < grid_size) & (indices[:, 1] >= 0) & (indices[:, 1] < grid_size)]
    #indices = np.clip(indices, 0, grid_size - 1)                #Clipping masses of particles off edges to nearest edge
    # Use np.add.at for proper accumulation
    np.add.at(density, (indices[:, 0], indices[:, 1], indices[:, 2]), masses) #1 for unit mass
    return density

#Cloud-in-Cell assignment

#Potential
def compute_potential(density, grid_size):
    k = np.fft.fftfreq(grid_size)
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    k2 = kx**2 + ky**2 + kz**2
    k2[0, 0, 0] = 1     #Division by 0 clause
    density_k = fftn(density)
    potential_k = density_k / k2
    potential_k[0, 0, 0] = 0.0  # Set mean of potential to zero (removes constant offset)
    potential = np.real(ifftn(potential_k))
    return potential

#Force
def interpolate_force(potential, positions, grid_size, box_size):
    grad_x = np.gradient(potential, axis=0)
    grad_y = np.gradient(potential, axis=1)
    grad_z = np.gradient(potential, axis=2)
    indices = (positions / box_size * grid_size).astype(int) % grid_size
    ix, iy, iz = indices[:, 0], indices[:, 1], indices[:, 2]
    # Interpolate force from gradients at particle grid locations
    forces = np.stack([grad_x[ix, iy, iz], grad_y[ix, iy, iz], grad_z[ix, iy, iz]], axis=1)
    return forces
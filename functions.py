import numpy as np
from scipy.fft import fft2, ifft2

#Nearest Grid Point assignment
def NGP(positions, grid_size, box_size, masses=None):
    density = np.zeros((grid_size, grid_size), dtype=float)
    # Normalize to grid indices
    indices = np.floor(positions / box_size * grid_size).astype(int)
    # Only use in-bounds indices
    indices_in = indices[(indices[:, 0] >= 0) & (indices[:, 0] < grid_size) & (indices[:, 1] >= 0) & (indices[:, 1] < grid_size)]
    #indices = np.clip(indices, 0, grid_size - 1)                #Clipping masses of particles off edges to nearest edge
    if masses is None:
        masses_in = np.ones(len(indices_in))
    else:
        masses_in = masses[indices_in]
    # Use np.add.at for proper accumulation
    np.add.at(density, (indices_in[:, 0], indices_in[:, 1]), masses_in) #1 for unit mass
    return density

#Cloud-in-Cell assignment

#Potential
def compute_potential(density, grid_size):
    kx = np.fft.fftfreq(grid_size).reshape(-1, 1)
    ky = np.fft.fftfreq(grid_size).reshape(1, -1)
    k2 = kx**2 + ky**2
    k2[0, 0] = 1
    density_k = fft2(density)
    potential_k = density_k / k2
    potential = np.real(ifft2(potential_k))
    return potential

#Force
def interpolate_force(potential, positions, grid_size, box_size):
    grad_x = np.gradient(potential, axis=0)
    grad_y = np.gradient(potential, axis=1)
    indices = (positions / box_size * grid_size).astype(int) % grid_size
    forces = np.stack([grad_x[indices[:,0], indices[:,1]], grad_y[indices[:,0], indices[:,1]]], axis=1)
    return forces
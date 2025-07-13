import numpy as np

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

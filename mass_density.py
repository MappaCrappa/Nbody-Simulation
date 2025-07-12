import numpy as np

#Nearest Grid Point assignment
def NGP(positions, grid_size, box_size):
    density = np.zeros((grid_size, grid_size))
    indices = (positions / box_size * grid_size).astype(int) % grid_size
    np.add.at(density, (indices[:,0], indices[:,1]), 1)
    return density

#Cloud-in-Cell assignment

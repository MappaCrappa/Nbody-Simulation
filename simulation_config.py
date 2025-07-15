import numpy as np

#Please input initial conditions and parameters for the simulation below.

#Simulation parameters (to be replaced with galaxy setup)
N_particles = 100
masses = np.ones(N_particles)
masses[0] = 5                    # Tracer particle mass
grid_size = 256                  # Taxing
box_size = 1.0
dt = 0.01
steps = 5000

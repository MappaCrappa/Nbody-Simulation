from galaxy_generation_functions import *

# Define your galaxy with generate_galaxy()
generate_galaxy("ellipse", "importance", 10000, a = 20.0, b = 12.0, c = 10.0, M_tot = 100000, seed=1052)

#generate_galaxy("diffuse_sphere", "importance", N_particles=5000, R = 10.0, M_tot = 2500, seed=1052)

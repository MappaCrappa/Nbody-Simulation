from galaxy_generation_functions import *

# Define your galaxy with generate_galaxy()
generate_galaxy("ellipse", "importance", 100000, a = 3, b = 1.8, c = 1.5, M_tot = 100000, seed=1052)
#generate_galaxy("diffuse_sphere", "importance", N_particles=100000, R = 3, M_tot = 100000, seed=42)

#generate_galaxy("diffuse_sphere", "importance", N_particles=100000, R = 1.25, M_tot = 1.55, seed=1052)

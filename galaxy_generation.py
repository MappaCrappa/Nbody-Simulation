from galaxy_generation_functions import *

# Define your galaxy with generate_galaxy()
#generate_galaxy("ellipse", "importance", 10000, a = 5.0, b = 3.0, c = 2.5, M_tot = 10000, seed=1052)

generate_galaxy("diffuse_sphere", "importance", 10000, R=10, M_tot = 10000, seed=1052)
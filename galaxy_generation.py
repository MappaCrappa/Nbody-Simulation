from galaxy_generation_functions import *

# Galaxy parameters
galaxy_type: str    = "disk"    # Type of galaxy
N: int              = 10000     # Number of Particles
M_tot: float        = 1.0       # Total mass
Rd, z0 = 1.0, 0.1
seed: int           = 42        # Seed for reproducability

# Equal-mass sampling from the target density
pos, vel, mass, meta = generate_disk_equal_mass(N, M_tot, Rd, z0, seed=seed)
save_galaxy_npz("Outputs/disk_equal_mass.npz", pos, mass, vel, meta)
view_configuration(pos, mass, title="Equal-mass sampling")


# Importance sampling: more uniform positions, masses scale with density
pos2, vel2, mass2, meta2 = generate_disk_importance_mass(N, M_tot, Rd, z0, max_mass_ratio=10, seed=seed)
save_galaxy_npz("Outputs/disk_importance_mass.npz", pos2, mass2, vel2, meta2)
view_configuration(pos2, mass2, title="Importance sampling")

"""
galaxy = generate_galaxy(
    "disk",
    N=50_000,
    M_tot=1.0,
    Rd=1.0,
    z0=0.1,
    sampling="equal",
    G=1.0,
    sigma_R=0.02,
    sigma_z=0.01,
    seed=42,
)

save_galaxy_npz("disk_equal.npz", galaxy)

# Preview
view_configuration(gal, s=0.2)
"""
#Supported Galaxy types and associated kwargs:


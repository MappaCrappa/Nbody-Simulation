from functions import *
import matplotlib.pyplot as plt

#Simulation parameters (to be replaced with galaxy setup)
grid_size = 200                        # Taxing
box_size = 10000.0
dt = 0.005
steps = 1000
CIC_interpolation = True              # NGP/CIC

#Configure logging
import logging as log
log_level = 'INFO'  # 'DEBUG', 'INFO', 'ERROR'
log.basicConfig(level=getattr(log, log_level), format='[%(levelname)s] %(message)s')    # Formatting
log.getLogger('matplotlib').setLevel(log.ERROR)                                         # Silence Matplotlib

#Tracking (to be added)

#Track Time
start_time = time.time()

# Import galaxies

# Add a function that chooses the galaxy to import with optional second galaxy and then the separation
positions, velocities, masses = (import_galaxy(
                  "Outputs/ellipse_importance_1052.npz",
                  "Outputs/diffuse_sphere_importance_1052.npz",
                        separation = 400.0,
                        direction = (1, 0, 0),
                        velocity = (0, 1000, 0)))

# Centre the simulation
positions += 0.5 * np.asarray(box_size, dtype=float) - positions.mean(axis=0)

# Collect positions and energy for visualisation
trajectory = []
energies = []

#Interpolation method
density_func = CIC if CIC_interpolation == True else NGP
force_func = force_CIC if CIC_interpolation == True else force_NGP

#Main loop
for step in range(steps):
    #Process info every 100th step
    if step % 100 == 0:
        log.info(f"Step {step}: Computing trajectories...")

    #Calculating density and potential
    density = density_func(positions, grid_size, box_size, masses)
    potential = compute_potential(density, grid_size)

    #Calculating forces (np.gradient outside the function to allow njit)
    potential_gradient = np.gradient(potential)
    forces = force_func(potential_gradient, positions, grid_size, box_size)

    #Updating particle kinematics
    velocities += forces * dt
    positions += velocities * dt
    positions = positions % box_size  # Apply periodic boundary
    trajectory.append(positions.copy())

    #Diagnostic Energy Tracker (every 10th step)
    if step % 10 == 0:
        KE = compute_kinetic_energy(velocities, masses)
        PE = compute_potential_energy(positions, masses, potential, grid_size, box_size)
        energies.append([KE, PE, KE + PE])

print(f"Main loop completed in {elapsed_time(start_time):.2f} seconds. Initialising visualisation...")

#Track Total Energy
energies = np.array(energies)
plt.figure()
plt.plot(energies[:,0], label='Kinetic')
plt.plot(energies[:,1], label='Potential')
plt.plot(energies[:,2], label='Total')
plt.xlabel('Step')
plt.ylabel('Energy')
plt.legend()
plt.show()

#Visualisation (Matplotlib_vis / Pyvista_mp4 / Pyvista_3D)
pyvista_mp4(trajectory, box_size)

print('Simulation and visualization complete!')
print(f"Simulation completed in {elapsed_time(start_time):.2f} seconds.")
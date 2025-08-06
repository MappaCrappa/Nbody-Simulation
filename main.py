from functions import *

#Please input initial conditions and parameters for the simulation below.

#Simulation parameters (to be replaced with galaxy setup)
N_particles = 500
tracer_mass = 5                       # Tracer particle mass
grid_size = 64                        # Taxing
box_size = 4.0
dt = 0.01
steps = 500
interpolation_method = 'CIC'   # NGP/CIC

#Configure logging
import logging as log
log_level = 'INFO'  # 'DEBUG' 'INFO', 'ERROR'
log.basicConfig(level=getattr(log, log_level), format='[%(levelname)s] %(message)s')    # Formatting
log.getLogger('matplotlib').setLevel(log.ERROR)                                         # Silence Matplotlib

#Tracking (to be added)

#Track Time
start_time = time.time()

# Initialize particle positions and (zero) velocities
centre = box_size / 2
spread = box_size / 16  # Smaller = more concentrated
positions = np.random.normal(loc=centre, scale=spread, size=(N_particles, 3))
velocities = np.zeros((N_particles, 3))
masses = np.ones(N_particles)
masses[0] = tracer_mass

# Collect positions for visualization
trajectory = []
energies = []

#Main loop
for step in range(steps):

    #Interpolation method choice from config. Change within function later
    if interpolation_method == 'NGP':
        density = NGP(positions, grid_size, box_size, masses)
    elif interpolation_method == 'CIC':
        density = CIC(positions, grid_size, box_size, masses)
    else:
        raise ValueError(f"Unknown interpolation method: {interpolation_method}")

    #Process info every 100th step
    if step % 100 == 0:
        log.info(f"Step {step}: Computing trajectories...")

    #Calculating potential and forces
    potential = compute_potential(density, grid_size)
    forces = force(potential, positions, grid_size, box_size, interpolation_method)

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

"""#Track Total Energy
energies = np.array(energies)
plt.figure()
plt.plot(energies[:,0], label='Kinetic')
plt.plot(energies[:,1], label='Potential')
plt.plot(energies[:,2], label='Total')
plt.xlabel('Step')
plt.ylabel('Energy')
plt.legend()
plt.show()"""

#Visualisation (Matplotlib_vis / Pyvista_mp4 / Pyvista_3D)
pyvista_mp4(trajectory, box_size)

print('Simulation and visualization complete!')
print(f"Simulation completed in {elapsed_time(start_time):.2f} seconds.")
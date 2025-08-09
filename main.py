from functions import *
import matplotlib.pyplot as plt

#Simulation parameters (to be replaced with galaxy setup)
N_particles = 500
tracer_mass = 5                       # Tracer particle mass
grid_size = 64                        # Taxing
box_size = 4.0
dt = 0.01
steps = 500
CIC_interpolation = True              # NGP/CIC

#Resolution settings (to be added)

#Configure logging
import logging as log
log_level = 'INFO'  # 'DEBUG', 'INFO', 'ERROR'
log.basicConfig(level=getattr(log, log_level), format='[%(levelname)s] %(message)s')    # Formatting
log.getLogger('matplotlib').setLevel(log.ERROR)                                         # Silence Matplotlib

#Tracking (to be added)

#Track Time
start_time = time.time()

# Initialize particle positions and (zero) velocities
centre = box_size / 2
spread = box_size / 16  # Smaller = more concentrated
positions = np.random.normal(loc=centre, scale=spread, size=(N_particles, 3)).astype(np.float32)
velocities = np.zeros((N_particles, 3), dtype=np.float32)
masses = np.ones(N_particles, dtype=np.float32)
masses[0] = tracer_mass

# Collect positions for visualization
trajectory = []
energies = []

#Interpolation method
density_func = CIC if CIC_interpolation == True else NGP
force_func = force_CIC if CIC_interpolation == True else NGP

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
print("Initializing visualization...")
pyvista_mp4(trajectory, box_size)

print('Simulation and visualization complete!')
print(f"Simulation completed in {elapsed_time(start_time):.2f} seconds.")
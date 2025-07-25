from functions import *

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

for step in range(steps):
    if interpolation_method == 'NGP':
        density = NGP(positions, grid_size, box_size, masses)
    elif interpolation_method == 'CIC':
        density = CIC(positions, grid_size, box_size, masses)
    else:
        raise ValueError(f"Unknown interpolation method: {interpolation_method}")
    if step % 100 == 0:
        log.info(f"Step {step}: Computing...")
    potential = compute_potential(density, grid_size)
    forces = force(potential, positions, grid_size, box_size)
    velocities += forces * dt
    positions += velocities * dt
    positions = positions % box_size  # Apply periodic boundary
    trajectory.append(positions.copy())
    if step % 10 == 0:  # Diagnostic Energy
        KE = compute_kinetic_energy(velocities, masses)
        PE = compute_potential_energy(positions, masses, potential, grid_size, box_size)
        energies.append([KE, PE, KE + PE])

print(f"Main loop completed in {elapsed_time(start_time):.2f} seconds.. Initialising visualisation...")

#Track Total Energy
energies = np.array(energies)
plt.figure()
plt.plot(energies[:,0], label='Kinetic')
plt.plot(energies[:,1], label='Potential')
plt.plot(energies[:,2], label='Total')
plt.xlabel('Diagnostic step')
plt.ylabel('Energy')
plt.legend()
plt.show()

#Visualisation (Matplotlib / Pyplot)
matplotlib_vis(trajectory, box_size)

print('Simulation and visualization complete!')
print(f"Simulation completed in {elapsed_time(start_time):.2f} seconds.")
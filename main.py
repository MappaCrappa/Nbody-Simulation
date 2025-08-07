from functions import *
import numpy as np

# Simulation parameters
N_particles = 5000
tracer_mass = 5
grid_size = 128
box_size = 40.0
dt = 0.01
steps = 500
interpolation_method = 'CIC'

# Configure logging
import logging as log
log_level = 'INFO'
log.basicConfig(level=getattr(log, log_level), format='[%(levelname)s] %(message)s')
log.getLogger('matplotlib').setLevel(log.ERROR)

# Pre-compute constants to avoid repeated calculations
cell_size = box_size / grid_size
inv_dt = 1.0 / dt
log_interval = 100
energy_interval = 10

# Track time
start_time = time.time()

# Initialize particle data
centre = box_size / 2
spread = box_size / 16
positions = np.random.normal(loc=centre, scale=spread, size=(N_particles, 3)).astype(np.float32)
velocities = np.zeros((N_particles, 3), dtype=np.float32)
masses = np.ones(N_particles, dtype=np.float32)
masses[0] = tracer_mass

# Pre-allocate arrays
trajectory_buffer = TrajectoryBuffer(steps, N_particles, compress=True)
energies = []
trajectory_buffer.append(positions.copy())  # Initial positions

# Pre-select interpolation method function to avoid if-else in loop
density_func = CIC_optimized if interpolation_method == 'CIC' else NGP

# Pre-allocate temporary arrays
temp_positions = np.empty_like(positions)
forces = np.empty_like(velocities)

print(f"Initialization completed. Starting main loop with {N_particles} particles...")

# Optimized main loop
for step in range(steps):
    # Progress logging (only when needed)
    if step % log_interval == 0:
        log.info(f"Step {step}/{steps}: Computing trajectories...")

    # Compute density (most expensive operation)
    density = density_func(positions, grid_size, box_size, masses)
    
    # Compute potential and forces
    potential = compute_potential_fft_optimized(density, grid_size)
    forces = compute_forces(potential, positions, grid_size, box_size, interpolation_method)

    # Update particle kinematics (vectorized operations)
    velocities += forces * dt
    temp_positions[:] = positions + velocities * dt
    np.mod(temp_positions, box_size, out=positions)  # In-place periodic boundary
    
    # Store trajectory (only copy when necessary)
    trajectory_buffer.append(positions)

    # Compute diagnostics less frequently
    if step % energy_interval == 0:
        KE = compute_kinetic_energy(velocities, masses)
        PE = compute_potential_energy(positions, masses, potential, grid_size, box_size)
        energies.append([KE, PE, KE + PE])

print(f"Main loop completed in {elapsed_time(start_time):.2f} seconds.")

# Optimized energy plotting
energies = np.array(energies)
plt.figure(figsize=(10, 6))
steps_energy = np.arange(0, len(energies) * energy_interval, energy_interval)
plt.plot(steps_energy, energies[:, 0], label='Kinetic', linewidth=2)
plt.plot(steps_energy, energies[:, 1], label='Potential', linewidth=2)
plt.plot(steps_energy, energies[:, 2], label='Total', linewidth=2)
plt.xlabel('Step')
plt.ylabel('Energy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Visualization
print("Initializing visualization...")
trajectory = trajectory_buffer.get_trajectory()
pyvista_mp4(trajectory, box_size)

print('Simulation and visualization complete!')
print(f"Total simulation time: {elapsed_time(start_time):.2f} seconds.")
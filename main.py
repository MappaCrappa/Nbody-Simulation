import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
import imageio as iio
from simulation_config import *
from functions import *

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
        potential = compute_potential(density, grid_size)
        forces = force_NGP(potential, positions, grid_size, box_size)
    elif interpolation_method == 'CIC':
        density = CIC(positions, grid_size, box_size, masses)
        potential = compute_potential(density, grid_size)
        forces = force_CIC(potential, positions, grid_size, box_size)
    else:
        raise ValueError(f"Unknown interpolation method: {interpolation_method}")
    if step % 100 == 0:
        log.debug(f"Step {step}: Computing...")
    velocities += forces * dt
    positions += velocities * dt
    positions = positions % box_size  # Apply periodic boundary
    trajectory.append(positions.copy())
    if step % 10 == 0:  # Diagnostic Energy
        KE = compute_kinetic_energy(velocities, masses)
        PE = compute_potential_energy(positions, masses, potential, grid_size, box_size)
        energies.append([KE, PE, KE + PE])

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

#np.save("Outputs/trajectory.npy", np.array(trajectory))
#np.save("Outputs/energies.npy", energies)

# Visualisation: Make GIF or MP4
fig = plt.figure(figsize=(8,8), dpi=80)
ax = fig.add_subplot(111, projection='3d')

# Get trajectory of the tracked particle
tracked_traj = np.array([frame[0] for frame in trajectory])
tail_length = 30  # or whatever you like
fade_min = 0.05   # Minimum opacity for the oldest segment

with iio.get_writer("Outputs/pm_nbody_sim.mp4", fps=20) as writer:
    for i, frame in enumerate(trajectory):
        ax.clear()
        ax.set_xlim(0, box_size)
        ax.set_ylim(0, box_size)
        ax.set_zlim(0, box_size)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.scatter(frame[:,0], frame[:,1], frame[:,2], s=1, color='black')
        ax.scatter(frame[0,0], frame[0,1], frame[0, 2], s=10, color='red')
        #Tail generation
        start = max(0, i - tail_length + 1)
        tail = tracked_traj[start:i + 1]
        n_tail = len(tail)
        if n_tail > 1:
            for j in range(n_tail - 1):
                alpha = fade_min + (1 - fade_min) * (j + 1) / n_tail
                ax.plot(tail[j:j + 2, 0], tail[j:j + 2, 1], tail[j:j + 2, 2], color='red', linewidth=1, alpha=alpha)
        fig.canvas.draw()
        image = np.array(fig.canvas.buffer_rgba())[..., :3]  # RGB image
        writer.append_data(image)
    writer.close()

print('Simulation and visualization complete!')
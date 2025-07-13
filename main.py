import numpy as np
from scipy.fft import fft2, ifft2
import matplotlib.pyplot as plt
import imageio

from simulation_config import *
import mass_density

# Initialize particle positions and (zero) velocities
centre = box_size / 2
spread = box_size / 8  # Smaller = more concentrated
positions = np.random.normal(loc=centre, scale=spread, size=(N_particles, 2))
velocities = np.zeros((N_particles, 2))

def compute_potential(density, grid_size):
    kx = np.fft.fftfreq(grid_size).reshape(-1, 1)
    ky = np.fft.fftfreq(grid_size).reshape(1, -1)
    k2 = kx**2 + ky**2
    k2[0, 0] = 1
    density_k = fft2(density)
    potential_k = density_k / k2
    potential = np.real(ifft2(potential_k))
    return potential

def interpolate_force(potential, positions, grid_size, box_size):
    grad_x = np.gradient(potential, axis=0)
    grad_y = np.gradient(potential, axis=1)
    indices = (positions / box_size * grid_size).astype(int) % grid_size
    forces = np.stack([grad_x[indices[:,0], indices[:,1]], grad_y[indices[:,0], indices[:,1]]], axis=1)
    return forces

# Collect positions for visualization
trajectory = []
COM = []            #Centre of Mass

for step in range(steps):
    density = mass_density.NGP(positions, grid_size, box_size, masses=None)
    potential = compute_potential(density, grid_size)
    forces = interpolate_force(potential, positions, grid_size, box_size)
    velocities += forces * dt
    positions += velocities * dt
    trajectory.append(positions.copy())
    #Track centre of mass
    com_array = np.average(positions, axis=0, weights=masses)
    COM.append(com_array.copy())

# Visualisation: Make GIF or MP4
fig, ax = plt.subplots(figsize=(16,16))
ims = []

# Get trajectory of the tracked particle
tracked_traj = np.array([frame[0] for frame in trajectory])
tail_length = 30  # or whatever you like
fade_min = 0.05   # Minimum opacity for the oldest segment

for i, frame in enumerate(trajectory):
    ax.clear()
    #ax.set_xlim(0, box_size)
    #ax.set_ylim(0, box_size)
    cm = COM[i]
    ax.set_xlim(cm[0] - box_size / 2, cm[0] + box_size / 2)
    ax.set_ylim(cm[1] - box_size / 2, cm[1] + box_size / 2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.scatter(frame[:,0], frame[:,1], s=1, color='black')
    ax.scatter(frame[0,0], frame[0,1], s=10, color='red')
    #Tail generation
    start = max(0, i - tail_length + 1)
    tail = tracked_traj[start:i + 1]
    n_tail = len(tail)
    if n_tail > 1:
        for j in range(n_tail - 1):
            alpha = fade_min + (1 - fade_min) * (j + 1) / n_tail
            ax.plot(tail[j:j + 2, 0], tail[j:j + 2, 1],
                    color='red', linewidth=1, alpha=alpha)

    fig.canvas.draw()
    image = np.array(fig.canvas.buffer_rgba())
    ims.append(image[..., :3])

#imageio.mimsave('Outputs/pm_nbody_sim.gif', ims, duration=0.05)                     # Save as GIF
imageio.mimsave('Outputs/pm_nbody_sim.mp4', ims, fps=20, macro_block_size=1)        # Save as MP4

print('Simulation and visualization complete! GIF/MP4 saved.')
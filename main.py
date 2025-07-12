import numpy as np
from scipy.fft import fft2, ifft2
import matplotlib.pyplot as plt
import imageio

#Simulation parameters (to be replaced with galaxy setup)
N_particles = 10000
grid_size = 640
box_size = 1.0
dt = 0.01
steps = 100

# Initialize particle positions and velocities
positions = np.random.rand(N_particles, 2) * box_size
velocities = np.zeros((N_particles, 2))

def assign_density(positions, grid_size, box_size):
    density = np.zeros((grid_size, grid_size))
    indices = (positions / box_size * grid_size).astype(int) % grid_size
    np.add.at(density, (indices[:,0], indices[:,1]), 1)
    return density

def compute_potential(density, grid_size):
    kx = np.fft.fftfreq(grid_size).reshape(-1, 1)
    ky = np.fft.fftfreq(grid_size).reshape(1, -1)
    k2 = kx**2 + ky**2
    k2[0, 0] = 1
    density_k = fft2(density)
    potential_k = -density_k / k2
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

for step in range(steps):
    density = assign_density(positions, grid_size, box_size)
    potential = compute_potential(density, grid_size)
    forces = interpolate_force(potential, positions, grid_size, box_size)
    velocities += forces * dt
    positions += velocities * dt
    positions %= box_size  # periodic boundary
    trajectory.append(positions.copy())

# Visualisation: Make GIF or MP4
fig, ax = plt.subplots(figsize=(5,5))
ims = []

for frame in trajectory:
    ax.clear()
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_xticks([])
    ax.set_yticks([])
    im = ax.scatter(frame[:,0], frame[:,1], s=1, color='black')
    fig.canvas.draw()
    image = np.array(fig.canvas.buffer_rgba())
    ims.append(image[..., :3])

imageio.mimsave('Outputs/pm_nbody_sim.gif', ims, duration=0.05)  # Save as GIF
#imageio.mimsave('pm_nbody_sim.mp4', ims, fps=20)         # Save as MP4

print("Simulation and visualization complete! GIF/MP4 saved.")
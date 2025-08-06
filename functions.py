import numpy as np
from numba import njit, prange
from scipy.fft import fftn, ifftn
#from simulation_config import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
import imageio as iio
import time
import pyvista as pv
from pyvistaqt import BackgroundPlotter
import threading

# Nearest Grid Point assignment
def NGP(positions, grid_size, box_size, masses):
    density = np.zeros((grid_size,)*3, dtype=float)
    positions = positions % box_size                                                                        # Wrapped positions across box edges
    indices = np.floor(positions / box_size * grid_size).astype(int)                                        # Normalize to grid indices
    np.add.at(density, (indices[:, 0], indices[:, 1], indices[:, 2]), masses)                        # Mass Counting
    return density

# Cloud-in-Cell assignment
@njit(parallel=True, fastmath=True)
def CIC(positions, grid_size, box_size, masses):
    density = np.zeros((grid_size, grid_size, grid_size), dtype=float)
    cell_size = box_size / grid_size
    for p in prange(positions.shape[0]):    #For Nth particle in Positions
        position = positions[p] % box_size
        mass = masses[p]
        # Find normalized cell indices and weights
        scaled_pos = position / cell_size
        i = np.floor(scaled_pos).astype(np.int64)
        d = scaled_pos - i
        for dx in (0, 1):
            wx = 1.0 - d[0] if dx == 0 else d[0]
            x = (i[0] + dx) % grid_size
            for dy in (0, 1):
                wy = 1.0 - d[1] if dy == 0 else d[1]
                y = (i[1] + dy) % grid_size
                for dz in (0, 1):
                    wz = 1.0 - d[2] if dz == 0 else d[2]
                    z = (i[2] + dz) % grid_size
                    density[x, y, z] += mass * wx * wy * wz
    return density

# Potential
def compute_potential(density, grid_size):
    k = np.fft.fftfreq(grid_size)
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    k2 = kx**2 + ky**2 + kz**2
    k2[0, 0, 0] = 1     #Division by 0 clause
    density_mean = np.mean(density) #make discrete kernal for accuracy to grid
    density_k = fftn(density - density_mean)
    potential_k = density_k / k2
    potential_k[0, 0, 0] = 0.0  # Set mean of potential to zero (removes constant offset)
    potential = np.real(ifftn(potential_k))
    return potential

#Force
def force(potential, positions, grid_size, box_size, interpolation_method):
    grad_x = np.gradient(potential, axis=0)
    grad_y = np.gradient(potential, axis=1)
    grad_z = np.gradient(potential, axis=2)
    if interpolation_method == 'NGP': # Interpolate force from gradients at particle grid locations
        indices = (positions / box_size * grid_size).astype(int) % grid_size
        ix, iy, iz = indices[:, 0], indices[:, 1], indices[:, 2]
        forces = np.stack([grad_x[ix, iy, iz], grad_y[ix, iy, iz], grad_z[ix, iy, iz]], axis=1)
    elif interpolation_method == 'CIC':
        force_grid = np.stack([grad_x, grad_y, grad_z], axis=-1)  # shape (Nx,Ny,Nz,3)
        N_particles = positions.shape[0]
        cell_size = box_size / grid_size
        forces = np.zeros((N_particles, 3))
        positions = positions % box_size

        scaled_pos = positions / cell_size
        i = np.floor(scaled_pos).astype(int)
        d = scaled_pos - i

        for p in range(N_particles):
            base = i[p]
            w = d[p]
            for dx in [0, 1]:
                wx = (1 - w[0]) if dx == 0 else w[0]
                x = (base[0] + dx) % grid_size
                for dy in [0, 1]:
                    wy = (1 - w[1]) if dy == 0 else w[1]
                    y = (base[1] + dy) % grid_size
                    for dz in [0, 1]:
                        wz = (1 - w[2]) if dz == 0 else w[2]
                        z = (base[2] + dz) % grid_size
                        weight = wx * wy * wz
                        forces[p] += force_grid[x, y, z] * weight
    else:
        raise ValueError(f"Unknown interpolation method: {interpolation_method}")
    return forces

def force_NGP(potential, positions, grid_size, box_size):   #Deprecated
    grad_x = np.gradient(potential, axis=0)
    grad_y = np.gradient(potential, axis=1)
    grad_z = np.gradient(potential, axis=2)
    indices = (positions / box_size * grid_size).astype(int) % grid_size
    ix, iy, iz = indices[:, 0], indices[:, 1], indices[:, 2]
    # Interpolate force from gradients at particle grid locations
    forces = np.stack([grad_x[ix, iy, iz], grad_y[ix, iy, iz], grad_z[ix, iy, iz]], axis=1)
    return forces

def force_CIC(potential, positions, grid_size, box_size):   #Deprecated
    grad_x = np.gradient(potential, axis=0)
    grad_y = np.gradient(potential, axis=1)
    grad_z = np.gradient(potential, axis=2)
    force_grid = np.stack([grad_x, grad_y, grad_z], axis=-1)  # shape (Nx,Ny,Nz,3)
    N_particles = positions.shape[0]
    cell_size = box_size / grid_size
    forces = np.zeros((N_particles, 3))
    positions = positions % box_size

    scaled_pos = positions / cell_size
    i = np.floor(scaled_pos).astype(int)
    d = scaled_pos - i

    for p in range(N_particles):
        base = i[p]
        w = d[p]
        for dx in [0, 1]:
            wx = (1 - w[0]) if dx == 0 else w[0]
            x = (base[0] + dx) % grid_size
            for dy in [0, 1]:
                wy = (1 - w[1]) if dy == 0 else w[1]
                y = (base[1] + dy) % grid_size
                for dz in [0, 1]:
                    wz = (1 - w[2]) if dz == 0 else w[2]
                    z = (base[2] + dz) % grid_size
                    weight = wx * wy * wz
                    forces[p] += force_grid[x, y, z] * weight
    return forces

#Visualisation
def matplotlib_vis(trajectory, box_size, output_path="Outputs/pm_nbody_sim.mp4", tail_length=30, fade_min=0.05, fps=20):
    fig = plt.figure(figsize=(8, 8), dpi=80)
    ax = fig.add_subplot(111, projection='3d')
    tracked_traj = np.array([frame[0] for frame in trajectory])
    with iio.get_writer("Outputs/pm_nbody_sim.mp4", fps=20) as writer:
        for i, frame in enumerate(trajectory):
            ax.clear()
            ax.set_xlim(0, box_size)
            ax.set_ylim(0, box_size)
            ax.set_zlim(0, box_size)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.scatter(frame[:, 0], frame[:, 1], frame[:, 2], s=1, color='black')
            ax.scatter(frame[0, 0], frame[0, 1], frame[0, 2], s=10, color='red')
            # Tail generation
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

def pyvista_mp4(trajectory, box_size, output_path="Outputs/pm_nbody_sim.mp4", fps=20):
    plotter = pv.Plotter(off_screen=True, window_size=(800, 800))
    plotter.set_background("black")

    # Particle Populations
    cloud = pv.PolyData(trajectory[0])
    plotter.add_points(cloud, color="white", point_size=2, render_points_as_spheres=True)

    tracer = pv.PolyData(trajectory[0][0:1])
    plotter.add_points(tracer, color="red", point_size=5, render_points_as_spheres=True)

    # Bounding Mesh
    bounds = pv.Cube(center=(box_size / 2, box_size / 2, box_size / 2), x_length=box_size, y_length=box_size, z_length=box_size)
    plotter.add_mesh(bounds, color='gray', style='wireframe', opacity=0.25, line_width=1)

    with iio.get_writer(output_path, fps=fps) as writer:
        for i, frame in enumerate(trajectory):
            cloud.points = frame                # Update all particles
            tracer.points = frame[0:1]          # Update tracer
            plotter.render()
            filename = f"Outputs/_frame_tmp.png"
            plotter.screenshot(filename)
            writer.append_data(iio.v2.imread(filename))
    plotter.close()

def pyvista_3D(trajectory, delay=20): #WIP non-functional
    plotter = BackgroundPlotter(window_size=(800, 800))
    plotter.set_background("black")

    # Particle Populations
    cloud = pv.PolyData(trajectory[0])
    plotter.add_points(cloud, color="white", point_size=2, render_points_as_spheres=True)

    tracer = pv.PolyData(trajectory[0][0:1])
    plotter.add_points(tracer, color="red", point_size=5, render_points_as_spheres=True)

    # Bounding Mesh
    bounds = pv.Cube(center=(box_size / 2, box_size / 2, box_size / 2), x_length=box_size, y_length=box_size, z_length=box_size)
    plotter.add_mesh(bounds, color='gray', style='wireframe', opacity=0.25, line_width=1)

    def animate():
        while not plotter._closed:
            for frame in trajectory:
                if plotter._closed:
                    break
                cloud.points = frame
                tracer.points = frame[0:1]
                plotter.update()
                time.sleep(delay)  # delay in ms

    # Run the animation in a background thread so you can rotate/zoom live!
    threading.Thread(target=animate, daemon=True).start()

#Diagnostic Functions
def compute_kinetic_energy(velocities, masses):
    KE = 0.5 * np.sum(masses * np.sum(velocities**2, axis=1))
    return KE

def compute_potential_energy(positions, masses, potential, grid_size, box_size):
    # Assign particle to nearest grid for potential
    indices = (positions / box_size * grid_size).astype(int) % grid_size
    phi_vals = potential[indices[:,0], indices[:,1], indices[:,2]]
    PE = np.sum(masses * phi_vals) * 0.5  # Factor 0.5 to avoid double-counting
    return PE

def elapsed_time(start_time):
    return time.time() - start_time
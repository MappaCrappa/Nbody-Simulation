import time
import numpy as np
from numba import njit
from scipy.fft import fftn, ifftn
import pyvista as pv
from pyvistaqt import BackgroundPlotter
import threading

# Nearest Grid Point assignment
def NGP(positions: np.ndarray, grid_size: int, box_size: float, masses: np.ndarray)-> np.ndarray:
    density = np.zeros((grid_size,)*3, dtype=float)                                                         # Intialising density value at 0
    positions = positions % box_size                                                                        # Wrapped positions across box edges
    indices = np.floor(positions / box_size * grid_size).astype(int)                                        # Normalize to grid indices
    np.add.at(density, (indices[:, 0], indices[:, 1], indices[:, 2]), masses)                        # Mass Counting
    return density

# Cloud-in-Cell assignment with minimal memory allocations (parallel=False is faster)
@njit(parallel=False, fastmath=True, cache=True)
def CIC(positions: np.ndarray, grid_size: int, box_size: float, masses: np.ndarray) -> np.ndarray:

    density = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    inverse_cell_size = grid_size / box_size  # Pre-compute inverse

    for p in range(positions.shape[0]):
        # Wrap positions efficiently
        x = positions[p, 0] % box_size
        y = positions[p, 1] % box_size
        z = positions[p, 2] % box_size

        # Scale to grid coordinates
        scaled_x = x * inverse_cell_size
        scaled_y = y * inverse_cell_size
        scaled_z = z * inverse_cell_size

        # Floor and fractional parts
        i = int(scaled_x)
        j = int(scaled_y)
        k = int(scaled_z)

        dx = scaled_x - i
        dy = scaled_y - j
        dz = scaled_z - k

        # Pre-compute weights
        wx0 = 1.0 - dx
        wx1 = dx
        wy0 = 1.0 - dy
        wy1 = dy
        wz0 = 1.0 - dz
        wz1 = dz

        mass = masses[p]

        # Unroll loops for better performance
        i1 = (i + 1) % grid_size
        j1 = (j + 1) % grid_size
        k1 = (k + 1) % grid_size

        # 8 corner contributions
        density[i, j, k] += mass * wx0 * wy0 * wz0
        density[i1, j, k] += mass * wx1 * wy0 * wz0
        density[i, j1, k] += mass * wx0 * wy1 * wz0
        density[i, j, k1] += mass * wx0 * wy0 * wz1
        density[i1, j1, k] += mass * wx1 * wy1 * wz0
        density[i1, j, k1] += mass * wx1 * wy0 * wz1
        density[i, j1, k1] += mass * wx0 * wy1 * wz1
        density[i1, j1, k1] += mass * wx1 * wy1 * wz1

    return density

# Potential
def compute_potential(density: np.ndarray, grid_size: int) -> np.ndarray:
    # Pre-computing k2 grid
    k = np.fft.fftfreq(grid_size).astype(np.float32)
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    k2 = kx**2 + ky**2 + kz**2
    k2[0, 0, 0] = 1             # Division by 0 clause

    # Fourier transform
    density_mean = np.mean(density)
    density_k = fftn(density - density_mean)

    # Poisson equation
    potential_k = density_k / k2
    potential_k[0, 0, 0] = 0.0  # Set mean of potential to zero (removes constant offset)

    #Inverse Fourier transform
    return np.real(ifftn(potential_k))

#Force
def force_NGP(potential_gradient: np.ndarray, positions: np.ndarray, grid_size: int, box_size: float) -> np.ndarray:
    grad_x, grad_y, grad_z = potential_gradient[0], potential_gradient[1], potential_gradient[2]
    indices = (positions / box_size * grid_size).astype(int) % grid_size
    ix, iy, iz = indices[:, 0], indices[:, 1], indices[:, 2]
    forces = np.stack([grad_x[ix, iy, iz], grad_y[ix, iy, iz], grad_z[ix, iy, iz]], axis=1)
    return forces

@njit(parallel=False, fastmath=True, cache=True)
def force_CIC(potential_gradient: np.ndarray, positions: np.ndarray, grid_size: int, box_size: float) -> np.ndarray:
    N_particles = positions.shape[0]
    forces = np.zeros((N_particles, 3), dtype=np.float32)
    inv_cell_size = grid_size / box_size
    grad_x, grad_y, grad_z = potential_gradient[0], potential_gradient[1], potential_gradient[2]

    for p in range(N_particles):
        #Position wrapping
        x = positions[p, 0] % box_size
        y = positions[p, 1] % box_size
        z = positions[p, 2] % box_size

        #Scale to grid coordinates
        scaled_x = x * inv_cell_size
        scaled_y = y * inv_cell_size
        scaled_z = z * inv_cell_size

        #Floor & fractional parts
        i = int(scaled_x)
        j = int(scaled_y)
        k = int(scaled_z)

        dx = scaled_x - i
        dy = scaled_y - j
        dz = scaled_z - k

        # Pre-compute weights
        wx0 = 1.0 - dx
        wx1 = dx
        wy0 = 1.0 - dy
        wy1 = dy
        wz0 = 1.0 - dz
        wz1 = dz

        # Unroll loops for better performance
        i1 = (i + 1) % grid_size
        j1 = (j + 1) % grid_size
        k1 = (k + 1) % grid_size

        # Accumulate forces from 8 corners
        w = wx0 * wy0 * wz0
        forces[p, 0] += grad_x[i, j, k] * w
        forces[p, 1] += grad_y[i, j, k] * w
        forces[p, 2] += grad_z[i, j, k] * w

        w = wx1 * wy0 * wz0
        forces[p, 0] += grad_x[i1, j, k] * w
        forces[p, 1] += grad_y[i1, j, k] * w
        forces[p, 2] += grad_z[i1, j, k] * w

        w = wx0 * wy1 * wz0
        forces[p, 0] += grad_x[i, j1, k] * w
        forces[p, 1] += grad_y[i, j1, k] * w
        forces[p, 2] += grad_z[i, j1, k] * w

        w = wx0 * wy0 * wz1
        forces[p, 0] += grad_x[i, j, k1] * w
        forces[p, 1] += grad_y[i, j, k1] * w
        forces[p, 2] += grad_z[i, j, k1] * w

        w = wx1 * wy1 * wz0
        forces[p, 0] += grad_x[i1, j1, k] * w
        forces[p, 1] += grad_y[i1, j1, k] * w
        forces[p, 2] += grad_z[i1, j1, k] * w

        w = wx1 * wy0 * wz1
        forces[p, 0] += grad_x[i1, j, k1] * w
        forces[p, 1] += grad_y[i1, j, k1] * w
        forces[p, 2] += grad_z[i1, j, k1] * w

        w = wx0 * wy1 * wz1
        forces[p, 0] += grad_x[i, j1, k1] * w
        forces[p, 1] += grad_y[i, j1, k1] * w
        forces[p, 2] += grad_z[i, j1, k1] * w

        w = wx1 * wy1 * wz1
        forces[p, 0] += grad_x[i1, j1, k1] * w
        forces[p, 1] += grad_y[i1, j1, k1] * w
        forces[p, 2] += grad_z[i1, j1, k1] * w

    return forces

#Visualisation
def pyvista_mp4(trajectory: list, box_size: float, output_path: str ="Outputs/pm_nbody_sim.mp4", fps: int=20):

    #Make trajectory a numpy array
    trajectory = np.asarray(trajectory)

    #Plotter configuration
    plotter = pv.Plotter(off_screen=True, window_size=(800, 800))
    plotter.set_background("black")

    # Particle Populations - Render_points_as_spheres = True has a bug on AMD GPUs on Windows where it produces no output -> Set to False
    cloud = pv.PolyData(trajectory[0])
    plotter.add_points(cloud, color="white", point_size=1, render_points_as_spheres=False)

    # Bounding Mesh
    bounds = pv.Cube(center=(box_size / 2, box_size / 2, box_size / 2), x_length=box_size, y_length=box_size, z_length=box_size)
    plotter.add_mesh(bounds, color='gray', style='wireframe', opacity=0.25, line_width=1)

    # Lock camera (saves recalculating)
    plotter.view_isometric()
    plotter.camera.zoom(8.0)

    #Write frames
    plotter.open_movie(output_path, framerate=fps)
    try:
        for i in range(trajectory.shape[0]):
            cloud.points[:] = trajectory[i]          # Update all particles
            plotter.write_frame()                    # Renders and appends the frame
    finally:
        plotter.close()

def pyvista_3D(trajectory, box_size, delay=20): #WIP non-functional
    plotter = BackgroundPlotter(window_size=(800, 800))
    plotter.set_background("black")

    # Particle Populations
    cloud = pv.PolyData(trajectory[0])
    plotter.add_points(cloud, color="white", point_size=1, render_points_as_spheres=True)

    # Bounding Mesh
    bounds = pv.Cube(center=(box_size / 2, box_size / 2, box_size / 2), x_length=box_size, y_length=box_size, z_length=box_size)
    plotter.add_mesh(bounds, color='gray', style='wireframe', opacity=0.25, line_width=1)

    # Zoom and focus
    plotter.camera.zoom(8.0)

    def animate():
        while not plotter._closed:
            for frame in trajectory:
                if plotter._closed:
                    break
                cloud.points = frame
                plotter.update()
                time.sleep(delay)  # delay in ms

    # Run the animation in a background thread so you can rotate/zoom live!
    threading.Thread(target=animate, daemon=True).start()

# Galaxy Importer
def import_galaxy(path1, path2=None, separation=1.0, direction=(1, 0, 0), velocity=(0, 0, 0)):
    with np.load(path1) as galaxy1_data:
        positions1 = galaxy1_data['pos'].astype(np.float32)
        velocities1 = galaxy1_data['vel'].astype(np.float32)
        masses1 = galaxy1_data['mass'].astype(np.float32)
    if path2 is None:
        return positions1, velocities1, masses1

    with np.load(path2) as galaxy2_data:
        positions2 = galaxy2_data['pos'].astype(np.float32)
        velocities2 = galaxy2_data['vel'].astype(np.float32)
        masses2 = galaxy2_data['mass'].astype(np.float32)

    # Centre of Mass
    com1 = (masses1[:, None] * positions1).sum(0) / masses1.sum()
    com2 = (masses2[:, None] * positions2).sum(0) / masses2.sum()

    # Place galaxy 2 so its COM is at com1 + sep_kpc * dir (positions in kpc)
    d = np.asarray(direction, float); d /= np.linalg.norm(d)
    target_com2 = com1 + d * float(separation)
    positions2 = positions2 + (target_com2 - com2)

    # Add velocity whilst ensuring zero total momentum (avoid drift)
    M1 = np.sum(masses1)
    M2 = np.sum(masses2)

    V1 = -(M2/(M1+M2))*np.asarray(velocity)
    V2 = (M1/(M1+M2))*np.asarray(velocity)

    print(V1, V2)

    velocities1 = velocities1 + np.asarray(V1)
    velocities2 = velocities2 + np.asarray(V2)

    #velocities2 = velocities2 + np.asarray(velocity)

    # combine and return like a simple import
    positions = np.vstack((positions1, positions2))
    velocities = np.vstack((velocities1, velocities2))
    masses   = np.concatenate((masses1, masses2))
    return positions, velocities, masses


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
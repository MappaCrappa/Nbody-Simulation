import time
import numpy as np
from numba import njit, prange, cuda
from scipy.fft import fftn, ifftn
import imageio as iio
import pyvista as pv
import matplotlib.pyplot as plt
from pyvistaqt import BackgroundPlotter
import threading

# Nearest Grid Point assignment
def NGP(positions, grid_size, box_size, masses):
    density = np.zeros((grid_size,)*3, dtype=float)                                                         # Intialising density value at 0
    positions = positions % box_size                                                                        # Wrapped positions across box edges
    indices = np.floor(positions / box_size * grid_size).astype(int)                                        # Normalize to grid indices
    np.add.at(density, (indices[:, 0], indices[:, 1], indices[:, 2]), masses)                        # Mass Counting
    return density

# Cloud-in-Cell assignment
@njit(parallel=False, fastmath=True)
def CIC(positions, grid_size, box_size, masses):
    density = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
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

@cuda.jit(parallel=True, fastmath=True)
def CIC_atomic(positions, grid_size, box_size, masses):
    Np = positions.shape[0]
    inv_cell = grid_size / box_size
    volume = (box_size / grid_size)**3

    # 1D density array for atomics
    dens_flat = np.zeros(grid_size*grid_size*grid_size, dtype=np.float64)

    for p in prange(Np):
        # wrap & scale to grid units
        xg = (positions[p,0] % box_size) * inv_cell
        yg = (positions[p,1] % box_size) * inv_cell
        zg = (positions[p,2] % box_size) * inv_cell

        i0 = int(np.floor(xg));  dx = xg - i0
        j0 = int(np.floor(yg));  dy = yg - j0
        k0 = int(np.floor(zg));  dz = zg - k0

        # the two grid-indices along each axis
        i1 = (i0 + 1) % grid_size
        j1 = (j0 + 1) % grid_size
        k1 = (k0 + 1) % grid_size

        # trilinear weights
        w000 = (1-dx)*(1-dy)*(1-dz)
        w100 =   dx *(1-dy)*(1-dz)
        w010 = (1-dx)*  dy *(1-dz)
        w001 = (1-dx)*(1-dy)*  dz
        w101 =   dx *(1-dy)*  dz
        w011 = (1-dx)*  dy *  dz
        w110 =   dx *  dy *(1-dz)
        w111 =   dx *  dy *  dz

        m = masses[p]
        # compute flat-indices = x + y⋅Nx + z⋅Nx Ny
        base0 = i0 + j0*grid_size + k0*grid_size*grid_size
        base1 = i1 + j0*grid_size + k0*grid_size*grid_size
        base2 = i0 + j1*grid_size + k0*grid_size*grid_size
        base3 = i0 + j0*grid_size + k1*grid_size*grid_size
        base4 = i1 + j0*grid_size + k1*grid_size*grid_size
        base5 = i0 + j1*grid_size + k1*grid_size*grid_size
        base6 = i1 + j1*grid_size + k0*grid_size*grid_size
        base7 = i1 + j1*grid_size + k1*grid_size*grid_size

        # atomic adds
        cuda.atomic.add(dens_flat, base0, m * w000)
        cuda.atomic.add(dens_flat, base1, m * w100)
        cuda.atomic.add(dens_flat, base2, m * w010)
        cuda.atomic.add(dens_flat, base3, m * w001)
        cuda.atomic.add(dens_flat, base4, m * w101)
        cuda.atomic.add(dens_flat, base5, m * w011)
        cuda.atomic.add(dens_flat, base6, m * w110)
        cuda.atomic.add(dens_flat, base7, m * w111)

    # reshape and convert to true mass density (mass per volume)
    density = dens_flat.reshape((grid_size,)*3)
    return density / volume

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

#Visualisation
def pyvista_mp4(trajectory, box_size, output_path="Outputs/pm_nbody_sim.mp4", fps=20):
    plotter = pv.Plotter(off_screen=True, window_size=(800, 800))
    plotter.set_background("black")

    # Particle Populations - Render_points_as_spheres = True has a bug on AMD GPUs on Windows where it produces no output -> Set to False
    cloud = pv.PolyData(trajectory[0])
    plotter.add_points(cloud, color="white", point_size=2, render_points_as_spheres=False)

    tracer = pv.PolyData(trajectory[0][0:1])
    plotter.add_points(tracer, color="red", point_size=5, render_points_as_spheres=False)

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

def pyvista_3D(trajectory, box_size, delay=20): #WIP non-functional
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
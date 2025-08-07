import time
import numpy as np
from numba import njit, prange, cuda
from scipy.fft import fftn, ifftn
import imageio as iio
import pyvista as pv
import matplotlib.pyplot as plt
from pyvistaqt import BackgroundPlotter
import threading
import os
from typing import Tuple, Optional, Union

# Nearest Grid Point assignment
@njit(parallel=True, fastmath=True)
def NGP(positions: np.ndarray, grid_size: int, box_size: float, masses: np.ndarray) -> np.ndarray:
    """
    Nearest Grid Point mass assignment scheme.
    
    Args:
        positions: Particle positions (N, 3)
        grid_size: Grid resolution
        box_size: Simulation box size
        masses: Particle masses (N,)
    
    Returns:
        density: Mass density on grid (grid_size, grid_size, grid_size)
    """
    density = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    positions = positions % box_size
    cell_size = box_size / grid_size
    
    for p in prange(positions.shape[0]):
        # Find grid indices
        indices = np.floor(positions[p] / cell_size).astype(np.int64)
        i = indices[0] % grid_size
        j = indices[1] % grid_size  
        k = indices[2] % grid_size
        density[i, j, k] += masses[p]
    
    return density

# Cloud-in-Cell assignment
@njit(fastmath=True)  # Remove parallel=True
def CIC(positions: np.ndarray, grid_size: int, box_size: float, masses: np.ndarray) -> np.ndarray:
    """
    Cloud-in-Cell mass assignment scheme (sequential, race-condition free).
    """
    density = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    cell_size = box_size / grid_size
    
    for p in range(positions.shape[0]):  # Sequential loop
        position = positions[p] % box_size
        mass = masses[p]
        
        scaled_pos = position / cell_size
        i = np.floor(scaled_pos).astype(np.int64)
        d = scaled_pos - i
        
        for dx in range(2):
            wx = 1.0 - d[0] if dx == 0 else d[0]
            x = (i[0] + dx) % grid_size
            for dy in range(2):
                wy = 1.0 - d[1] if dy == 0 else d[1]
                y = (i[1] + dy) % grid_size
                for dz in range(2):
                    wz = 1.0 - d[2] if dz == 0 else d[2]
                    z = (i[2] + dz) % grid_size
                    density[x, y, z] += mass * wx * wy * wz
    
    return density

@cuda.jit
def CIC_cuda_kernel(positions, masses, density_flat, grid_size, box_size):
    """CUDA kernel for CIC mass assignment."""
    p = cuda.grid(1)
    if p >= positions.shape[0]:
        return
        
    inv_cell = grid_size / box_size
    
    # Wrap & scale to grid units
    xg = (positions[p, 0] % box_size) * inv_cell
    yg = (positions[p, 1] % box_size) * inv_cell  
    zg = (positions[p, 2] % box_size) * inv_cell

    i0 = int(xg);  dx = xg - i0
    j0 = int(yg);  dy = yg - j0
    k0 = int(zg);  dz = zg - k0

    # Neighboring indices
    i1 = (i0 + 1) % grid_size
    j1 = (j0 + 1) % grid_size
    k1 = (k0 + 1) % grid_size

    # Trilinear weights
    weights = [
        (1-dx) * (1-dy) * (1-dz),  # w000
        dx * (1-dy) * (1-dz),      # w100
        (1-dx) * dy * (1-dz),      # w010
        (1-dx) * (1-dy) * dz,      # w001
        dx * (1-dy) * dz,          # w101
        (1-dx) * dy * dz,          # w011
        dx * dy * (1-dz),          # w110
        dx * dy * dz               # w111
    ]
    
    # Flat indices
    indices = [
        i0 + j0*grid_size + k0*grid_size*grid_size,
        i1 + j0*grid_size + k0*grid_size*grid_size,
        i0 + j1*grid_size + k0*grid_size*grid_size,
        i0 + j0*grid_size + k1*grid_size*grid_size,
        i1 + j0*grid_size + k1*grid_size*grid_size,
        i0 + j1*grid_size + k1*grid_size*grid_size,
        i1 + j1*grid_size + k0*grid_size*grid_size,
        i1 + j1*grid_size + k1*grid_size*grid_size
    ]

    mass = masses[p]
    for i, (idx, w) in enumerate(zip(indices, weights)):
        cuda.atomic.add(density_flat, idx, mass * w)

def CIC_cuda(positions: np.ndarray, grid_size: int, box_size: float, masses: np.ndarray) -> np.ndarray:
    """
    CUDA-accelerated Cloud-in-Cell mass assignment.
    
    Args:
        positions: Particle positions (N, 3)
        grid_size: Grid resolution
        box_size: Simulation box size
        masses: Particle masses (N,)
    
    Returns:
        density: Mass density on grid (grid_size, grid_size, grid_size)
    """
    N_particles = positions.shape[0]
    volume = (box_size / grid_size)**3
    
    # GPU arrays
    d_positions = cuda.to_device(positions.astype(np.float32))
    d_masses = cuda.to_device(masses.astype(np.float32))
    d_density_flat = cuda.device_array(grid_size**3, dtype=np.float32)
    
    # Launch kernel
    threads_per_block = 256
    blocks_per_grid = (N_particles + threads_per_block - 1) // threads_per_block
    CIC_cuda_kernel[blocks_per_grid, threads_per_block](
        d_positions, d_masses, d_density_flat, grid_size, box_size
    )
    
    # Get result and reshape
    density_flat = d_density_flat.copy_to_host()
    density = density_flat.reshape((grid_size, grid_size, grid_size))
    
    return density / volume

def compute_potential(density: np.ndarray, grid_size: int) -> np.ndarray:
    """
    Compute gravitational potential using FFT-based method (original behavior).
    """
    k = np.fft.fftfreq(grid_size)
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    k2 = kx**2 + ky**2 + kz**2
    k2[0, 0, 0] = 1     # Division by 0 clause (original approach)
    
    density_mean = np.mean(density)
    density_k = fftn(density - density_mean)
    potential_k = density_k / k2
    potential_k[0, 0, 0] = 0.0  # Set mean of potential to zero
    potential = np.real(ifftn(potential_k))
    return potential

@njit(parallel=True, fastmath=True)
def force_CIC_numba(grad_x: np.ndarray, grad_y: np.ndarray, grad_z: np.ndarray, 
                   positions: np.ndarray, grid_size: int, box_size: float) -> np.ndarray:
    """
    Numba-optimized CIC force interpolation.
    """
    N_particles = positions.shape[0]
    forces = np.zeros((N_particles, 3), dtype=np.float32)
    cell_size = box_size / grid_size
    
    for p in prange(N_particles):
        position = positions[p] % box_size
        scaled_pos = position / cell_size
        i = np.floor(scaled_pos).astype(np.int64)
        d = scaled_pos - i
        
        force_p = np.zeros(3, dtype=np.float32)
        
        for dx in range(2):
            wx = 1.0 - d[0] if dx == 0 else d[0]
            x = (i[0] + dx) % grid_size
            for dy in range(2):
                wy = 1.0 - d[1] if dy == 0 else d[1]
                y = (i[1] + dy) % grid_size
                for dz in range(2):
                    wz = 1.0 - d[2] if dz == 0 else d[2]
                    z = (i[2] + dz) % grid_size
                    weight = wx * wy * wz
                    
                    force_p[0] += grad_x[x, y, z] * weight
                    force_p[1] += grad_y[x, y, z] * weight
                    force_p[2] += grad_z[x, y, z] * weight
        
        forces[p] = force_p
    
    return forces

def compute_forces(potential: np.ndarray, positions: np.ndarray, grid_size: int, 
                  box_size: float, interpolation_method: str = 'CIC') -> np.ndarray:
    """
    Compute forces on particles (original scaling).
    """
    # Original gradient calculation without extra scaling
    grad_x = np.gradient(potential, axis=0)
    grad_y = np.gradient(potential, axis=1)
    grad_z = np.gradient(potential, axis=2)
    
    if interpolation_method == 'NGP':
        indices = (positions / box_size * grid_size).astype(int) % grid_size
        ix, iy, iz = indices[:, 0], indices[:, 1], indices[:, 2]
        forces = np.stack([grad_x[ix, iy, iz], grad_y[ix, iy, iz], grad_z[ix, iy, iz]], axis=1)
        
    elif interpolation_method == 'CIC':
        forces = force_CIC_numba(grad_x, grad_y, grad_z, positions, grid_size, box_size)
        
    else:
        raise ValueError(f"Unknown interpolation method: {interpolation_method}")
    
    return forces

def create_output_directory(path: str = "Outputs") -> None:
    """Create output directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def pyvista_mp4(trajectory: list, box_size: float, output_path: str = "Outputs/pm_nbody_sim.mp4", 
                fps: int = 20, point_size: int = 2, tracer_size: int = 5) -> None:
    """
    Create MP4 visualization of particle trajectory using PyVista.
    
    Args:
        trajectory: List of position arrays for each timestep
        box_size: Simulation box size
        output_path: Output video file path
        fps: Frames per second
        point_size: Size of regular particles
        tracer_size: Size of tracer particle
    """
    create_output_directory(os.path.dirname(output_path))
    
    plotter = pv.Plotter(off_screen=True, window_size=(800, 800))
    plotter.set_background("black")

    # Particle populations
    cloud = pv.PolyData(trajectory[0])
    plotter.add_points(cloud, color="white", point_size=point_size, render_points_as_spheres=False)

    tracer = pv.PolyData(trajectory[0][:1])
    plotter.add_points(tracer, color="red", point_size=tracer_size, render_points_as_spheres=False)

    # Bounding box
    bounds = pv.Cube(
        center=(box_size / 2, box_size / 2, box_size / 2),
        x_length=box_size, y_length=box_size, z_length=box_size
    )
    plotter.add_mesh(bounds, color='gray', style='wireframe', opacity=0.25, line_width=1)

    temp_frame_path = os.path.join(os.path.dirname(output_path), "_frame_tmp.png")
    
    with iio.get_writer(output_path, fps=fps) as writer:
        for i, frame in enumerate(trajectory):
            cloud.points = frame
            tracer.points = frame[:1]
            plotter.render()
            plotter.screenshot(temp_frame_path)
            writer.append_data(iio.v2.imread(temp_frame_path))
            
            if i % 50 == 0:
                print(f"Rendered frame {i}/{len(trajectory)}")
    
    # Cleanup
    if os.path.exists(temp_frame_path):
        os.remove(temp_frame_path)
    plotter.close()
    print(f"Video saved to {output_path}")

def pyvista_interactive(trajectory: list, box_size: float, delay: float = 0.05) -> None:
    """
    Interactive 3D visualization using PyVista (improved version).
    
    Args:
        trajectory: List of position arrays for each timestep
        box_size: Simulation box size  
        delay: Delay between frames in seconds
    """
    plotter = BackgroundPlotter(window_size=(800, 800))
    plotter.set_background("black")

    # Particle populations
    cloud = pv.PolyData(trajectory[0])
    actor_cloud = plotter.add_points(cloud, color="white", point_size=2, render_points_as_spheres=True)

    tracer = pv.PolyData(trajectory[0][:1])
    actor_tracer = plotter.add_points(tracer, color="red", point_size=5, render_points_as_spheres=True)

    # Bounding box
    bounds = pv.Cube(
        center=(box_size / 2, box_size / 2, box_size / 2),
        x_length=box_size, y_length=box_size, z_length=box_size
    )
    plotter.add_mesh(bounds, color='gray', style='wireframe', opacity=0.25, line_width=1)

    # Animation control
    animation_running = [True]  # Use list for mutable reference
    current_frame = [0]

    def animate():
        while not plotter._closed and animation_running[0]:
            if current_frame[0] >= len(trajectory):
                current_frame[0] = 0  # Loop animation
                
            frame = trajectory[current_frame[0]]
            cloud.points = frame
            tracer.points = frame[:1]
            plotter.update()
            current_frame[0] += 1
            time.sleep(delay)

    # Add controls
    def toggle_animation():
        animation_running[0] = not animation_running[0]
        if animation_running[0]:
            threading.Thread(target=animate, daemon=True).start()

    # Start animation
    threading.Thread(target=animate, daemon=True).start()
    print("Interactive visualization started. Close window to exit.")
    return plotter

# Diagnostic Functions
def compute_kinetic_energy(velocities: np.ndarray, masses: np.ndarray) -> float:
    """Compute total kinetic energy."""
    return 0.5 * np.sum(masses * np.sum(velocities**2, axis=1))

def compute_potential_energy(positions: np.ndarray, masses: np.ndarray, 
                           potential: np.ndarray, grid_size: int, box_size: float) -> float:
    """Compute total potential energy."""
    indices = (positions / box_size * grid_size).astype(int) % grid_size
    phi_vals = potential[indices[:, 0], indices[:, 1], indices[:, 2]]
    return 0.5 * np.sum(masses * phi_vals)

def compute_center_of_mass(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Compute center of mass."""
    return np.average(positions, axis=0, weights=masses)

def compute_virial_ratio(kinetic_energy: float, potential_energy: float) -> float:
    """Compute virial ratio (2T/|U|)."""
    return 2 * kinetic_energy / abs(potential_energy) if potential_energy != 0 else 0

def elapsed_time(start_time: float) -> float:
    """Return elapsed time since start_time."""
    return time.time() - start_time

class Timer:
    """Context manager for timing code blocks."""
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        print(f"{self.description} completed in {elapsed:.2f} seconds")

# Add these optimized functions to your functions.py

@njit(fastmath=True, cache=True)
def CIC_optimized(positions: np.ndarray, grid_size: int, box_size: float, masses: np.ndarray) -> np.ndarray:
    """
    Highly optimized CIC with minimal memory allocations.
    """
    density = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    inv_cell_size = grid_size / box_size  # Pre-compute inverse
    
    for p in range(positions.shape[0]):
        # Wrap positions efficiently
        x = positions[p, 0] % box_size
        y = positions[p, 1] % box_size
        z = positions[p, 2] % box_size
        
        # Scale to grid coordinates
        scaled_x = x * inv_cell_size
        scaled_y = y * inv_cell_size
        scaled_z = z * inv_cell_size
        
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

@njit(fastmath=True, cache=True)
def compute_potential_optimized(density: np.ndarray, grid_size: int) -> np.ndarray:
    """
    Optimized potential computation with minimal allocations.
    """
    # Pre-compute k arrays once
    k_1d = np.arange(grid_size, dtype=np.float32)
    k_1d[grid_size//2:] -= grid_size
    k_1d /= grid_size
    
    # Use broadcasting for k² calculation
    k2 = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                k2[i, j, k] = k_1d[i]**2 + k_1d[j]**2 + k_1d[k]**2
    
    k2[0, 0, 0] = 1.0  # Avoid division by zero
    
    return k2

def compute_potential_fft_optimized(density: np.ndarray, grid_size: int) -> np.ndarray:
    """
    Memory-efficient FFT-based potential computation.
    """
    # Use scipy's more efficient FFT planning
    density_mean = np.mean(density)
    density_centered = density - density_mean
    
    # Pre-compute k² grid (do this once and cache if possible)
    k = np.fft.fftfreq(grid_size).astype(np.float32)  # Convert to float32 after creation
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    k2 = kx*kx + ky*ky + kz*kz
    k2[0, 0, 0] = 1.0
    
    # Forward FFT
    density_k = fftn(density_centered)
    
    # Solve Poisson equation
    potential_k = density_k / k2
    potential_k[0, 0, 0] = 0.0
    
    # Inverse FFT
    return np.real(ifftn(potential_k))

@njit(parallel=False, fastmath=True, cache=True)  
def force_CIC_optimized(grad_x: np.ndarray, grad_y: np.ndarray, grad_z: np.ndarray,
                       positions: np.ndarray, grid_size: int, box_size: float) -> np.ndarray:
    """
    Optimized force interpolation with reduced memory access.
    """
    N_particles = positions.shape[0]
    forces = np.zeros((N_particles, 3), dtype=np.float32)
    inv_cell_size = grid_size / box_size
    
    for p in range(N_particles):
        x = positions[p, 0] % box_size
        y = positions[p, 1] % box_size  
        z = positions[p, 2] % box_size
        
        scaled_x = x * inv_cell_size
        scaled_y = y * inv_cell_size
        scaled_z = z * inv_cell_size
        
        i = int(scaled_x)
        j = int(scaled_y)
        k = int(scaled_z)
        
        dx = scaled_x - i
        dy = scaled_y - j
        dz = scaled_z - k
        
        wx0 = 1.0 - dx; wx1 = dx
        wy0 = 1.0 - dy; wy1 = dy
        wz0 = 1.0 - dz; wz1 = dz
        
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

# Memory-efficient trajectory storage
class TrajectoryBuffer:
    """Efficient trajectory storage with optional compression."""
    
    def __init__(self, max_frames: int, N_particles: int, compress: bool = True):
        self.max_frames = max_frames
        self.compress = compress
        self.frames = []
        self.frame_count = 0
        
    def append(self, positions: np.ndarray):
        if self.compress and self.frame_count > 0:
            # Store only every Nth frame for large simulations
            if self.frame_count % max(1, self.max_frames // 200) == 0:
                self.frames.append(positions.copy().astype(np.float32))
        else:
            self.frames.append(positions.copy().astype(np.float32))
        self.frame_count += 1
        
    def get_trajectory(self):
        return self.frames
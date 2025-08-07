import time
import numpy as np
from numba import njit
from scipy.fft import fftn, ifftn
import imageio as iio
import pyvista as pv
from pyvistaqt import BackgroundPlotter
import threading
import os

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
        forces = force_CIC_optimized(grad_x, grad_y, grad_z, positions, grid_size, box_size)
        
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
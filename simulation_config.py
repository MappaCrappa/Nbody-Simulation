#Please input initial conditions and parameters for the simulation below.

#Simulation parameters (to be replaced with galaxy setup)
N_particles = 250
tracer_mass = 5                       # Tracer particle mass
grid_size = 128                       # Taxing
box_size = 4.0
dt = 0.01
steps = 500
interpolation_method = 'CIC'   # NGP/CIC

#Configure logging
import logging as log
log_level = 'INFO'  # 'DEBUG' 'INFO', 'ERROR'
log.basicConfig(level=getattr(log, log_level), format='[%(levelname)s] %(message)s')    # Formatting
log.getLogger('matplotlib').setLevel(log.ERROR)                                         # Silence Matplotlib

#Tracking
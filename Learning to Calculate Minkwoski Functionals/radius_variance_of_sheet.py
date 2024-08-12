import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import jit
from quantimpy import minkowski as mk

# Parameters
volume_size = 5  # Number of sub-cubes along each dimension
sub_cube_size = 20  # Size of each sub-cube in the volume
ball_radius_list = [0.5 + i * 0.5 for i in range(50)]  # List of ball radii to test

def add_balls(volume, sub_cube_size, ball_radius):
    """
    Add spherical balls to the 3D volume.

    Parameters:
    - volume: 3D numpy array representing the volume
    - sub_cube_size: Size of each sub-cube
    - ball_radius: Radius of the balls to be added

    Returns:
    - Updated volume with balls added
    """
    # Create a grid of coordinates for the entire volume
    z, y, x = np.indices(volume.shape)

    # Compute the centers of each sub-cube in the volume
    centers = np.array([
        (i * sub_cube_size + sub_cube_size // 2, 
         j * sub_cube_size + sub_cube_size // 2, 
         k * sub_cube_size + sub_cube_size // 2) 
        for i in range(volume_size) 
        for j in range(volume_size) 
        for k in range(volume_size)
    ])

    # For each center, mark the volume within the ball radius as True
    for center_x, center_y, center_z in centers:
        # Compute distance from each point in the volume to the center
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2)
        # Set the points within the ball radius to True
        volume[dist_from_center <= ball_radius] = True

    return volume

def compute_minkowski_functionals(ball_radius):
    """
    Create a volume with balls and compute the Minkowski functionals.

    Parameters:
    - ball_radius: Radius of the balls to be added

    Returns:
    - Minkowski functionals computed for the volume
    """
    # Initialize a 3D volume with all False values
    volume_shape = (volume_size * sub_cube_size, volume_size * sub_cube_size, volume_size * sub_cube_size)
    volume = np.zeros(volume_shape, dtype=bool)

    # Add balls to the volume
    add_balls(volume, sub_cube_size, ball_radius)
    
    # Compute the Minkowski functionals for the volume
    minkowski_functionals = mk.functionals(volume)

    return minkowski_functionals

# Initialize lists to store Minkowski functionals for each ball radius
minkowski_functionals = [[] for _ in range(4)]  # Assuming there are 4 functionals

# Compute Minkowski functionals for each radius in the list
for ball_radius in tqdm(ball_radius_list, desc="Computing Minkowski Functionals"):
    functionals = compute_minkowski_functionals(ball_radius)  # Get all four functionals
    for i in range(4):
        minkowski_functionals[i].append(functionals[i])

# Plotting the Minkowski functionals in separate subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # Create a 2x2 grid of subplots

# Titles for each subplot
titles = ['Functional 1', 'Functional 2', 'Functional 3', 'Functional 4']

# Plot each Minkowski functional
for i, ax in enumerate(axs.flat):
    ax.plot(ball_radius_list, minkowski_functionals[i], marker='o')
    ax.set_xlabel('Ball Radius')
    ax.set_ylabel(f'Minkowski Functional {i + 1}')
    ax.set_title(titles[i])

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()
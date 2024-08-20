import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import jit
from quantimpy import minkowski as mk

# Parameters
volume_size = 3  # Number of sub-cubes along each dimension
sub_cube_size = 20   # Size of each sub-cube in the volume
ball_radius_list = [0.5 + i * 0.5 for i in range(200)]  # List of ball radii to test

def add_balls(volume, sub_cube_size, ball_radius):
    """
    Add spherical balls to the 3D volume.

    Parameters:
    - volume: 3D torch tensor representing the volume
    - sub_cube_size: Size of each sub-cube
    - ball_radius: Radius of the balls to be added

    Returns:
    - Updated volume with balls added
    """
    # Create a grid of coordinates for the entire volume
    z, y, x = torch.meshgrid(
        torch.arange(volume.shape[0], device=volume.device),
        torch.arange(volume.shape[1], device=volume.device),
        torch.arange(volume.shape[2], device=volume.device),
        indexing='ij'
    )

    # Compute the centers of each sub-cube in the volume
    volume_size = volume.shape[0] // sub_cube_size
    centers = torch.stack(
        [
            torch.arange(sub_cube_size // 2, volume_size * sub_cube_size, sub_cube_size, device=volume.device),
            torch.arange(sub_cube_size // 2, volume_size * sub_cube_size, sub_cube_size, device=volume.device),
            torch.arange(sub_cube_size // 2, volume_size * sub_cube_size, sub_cube_size, device=volume.device)
        ],
        dim=1
    ).view(-1, 3)

    # For each center, mark the volume within the ball radius as True
    for center_x, center_y, center_z in centers:
        center_x = int(center_x.item())
        center_y = int(center_y.item())
        center_z = int(center_z.item())
        
        # Compute distance from each point in the volume to the center
        dist_from_center = torch.sqrt((x - center_x).float()**2 + (y - center_y).float()**2 + (z - center_z).float()**2)
        # Set the points within the ball radius to True
        volume[dist_from_center <= ball_radius] = True

    return volume

def compute_minkowski_functionals(ball_radius):
    """
    Create a volume with balls and compute the Minkowski functionals.

    Parameters:
    - ball_radius: Radius of the balls to be added
    - volume_size: Size of the volume (number of sub-cubes)
    - sub_cube_size: Size of each sub-cube

    Returns:
    - Minkowski functionals computed for the volume
    """
    # Initialize a 3D volume with all False values
    volume_shape = (volume_size * sub_cube_size, volume_size * sub_cube_size, volume_size * sub_cube_size)
    volume = torch.zeros(volume_shape, dtype=torch.bool, device='cpu')

    # Add balls to the volume
    volume = add_balls(volume, sub_cube_size, ball_radius)
    
    # Assuming `mk.functionals` is a function that can handle PyTorch tensors, use it here.
    # If not, you might need to implement a PyTorch-compatible version.
    minkowski_functionals = mk.functionals(volume.numpy())

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
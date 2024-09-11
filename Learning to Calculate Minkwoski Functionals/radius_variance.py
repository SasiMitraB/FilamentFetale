"""
Script Name: radius_variance.py
Description:
    This script generates 3D volumes with spherical regions (balls) and computes Minkowski functionals
    for varying ball radii. The script visualizes each 3D volume and plots the computed functionals
    against ball radii.

    Key steps include:
    1. Generating a 3D volume with balls of different radii.
    2. Computing Minkowski functionals for the generated volume.
    3. Plotting the Minkowski functionals as a function of ball radius.

    When run, the script will:
    - Print the randomly generated center coordinates for the balls.
    - Compute and plot the Minkowski functionals (up to four) as a function of ball radius.
    - Display 2D plots of each functional in a 2x2 grid of subplots.

Usage:
    To execute this script, run it with Python from the command line:
    python radius_variance.py

Requirements:
    Ensure the following Python packages are installed:
    - numpy
    - matplotlib
    - tqdm
    - numba
    - quantimpy

Notes:
    - The script generates a random center for the balls within a 200x200x200 grid.
    - It tests ball radii from 3 to 75 in steps of 3.
    - The `quantimpy` library's `minkowski` module is used to compute Minkowski functionals.

Date:
    August 22, 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import jit
from quantimpy import minkowski as mk

# Parameters
grid_dimensions = 200 #Number of grid points along each dimension
ball_radius_list = [3 + i * 3 for i in range(25)]  # List of ball radii to test
#Randomly Generated Center in the grid for the ball
center = np.random.randint(0, grid_dimensions, size=3)
print(center)


def add_balls(center, radius):
    """
    Takes input of the center and radius, adds a ball at the specified center and radius, and returns the volume.

    Parameters:
    - center (tuple): The coordinates of the center of the ball in the form (x, y, z).
    - radius (float): The radius of the ball.

    Returns:
    - volume (ndarray): A 3D numpy array with dimensions (grid_dimensions, grid_dimensions, grid_dimensions).
                       The array represents a volume where everything is marked as False except for the positions
                       within the ball, which are marked as True.

    """
    
    volume = np.zeros((grid_dimensions, grid_dimensions, grid_dimensions), dtype=bool)
    x, y, z = center
    r = radius
    # Create a grid of coordinates
    coords = np.indices((grid_dimensions, grid_dimensions, grid_dimensions))
    
    # Calculate the squared distance from the center (x, y, z)
    squared_distances = (coords[0] - x) ** 2 + (coords[1] - y) ** 2 + (coords[2] - z) ** 2
    
    # Create a mask where the squared distance is less than or equal to r^2
    mask = squared_distances <= r ** 2
    
    # Apply the mask to the volume array
    volume[mask] = True
    return volume

def compute_minkowski_functionals(ball_radius):
    """
    Create a volume with balls and compute the Minkowski functionals.

    Parameters:
    - ball_radius: Radius of the balls to be added

    Returns:
    - Minkowski functionals computed for the volume
    """
    

    # Add balls to the volume
    volume = add_balls(center, ball_radius)
    
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
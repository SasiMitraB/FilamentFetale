import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import jit
from quantimpy import minkowski as mk
import concurrent.futures


def calculate_shapefinders(minkowski_functionals):
    """
    Calculates the shapefinders based on the given Minkowski functionals.
    Reference:
    - Section 2.5.1 of Dr Amit's thesis
    Parameters:
    - minkowski_functionals (list): A list of four Minkowski functionals [v_0, v_1, v_2, v_3].
    Returns:
    - thickness (float): The thickness shapefinder.
    - width (float): The width shapefinder.
    - length (float): The length shapefinder.
    """

    v_0, v_1, v_2, v_3 = minkowski_functionals

    l1 = v_0 / (2 * v_1)
    l2 = 2*v_1 / (np.pi * v_2)
    l3 = 3 * v_2 / (4 * v_3)

    # Sorting the shapefinders in descending order
    shapefinders = [l1, l2, l3]
    shapefinders.sort()
    thickness, width, length = shapefinders

    return thickness, width, length

@jit(nopython=True)
def planarity(thickness, width):
    """
    Calculates the planarity of an object based on its thickness and width.
    Parameters:
    thickness (float): The thickness of the object.
    width (float): The width of the object.
    Returns:
    float: The planarity value, which is a measure of how flat the object is.
    Reference:
    - Section 2.5.1 of Dr Amit's thesis
    """

    p = (width - thickness) / (width + thickness)
    return p

@jit(nopython=True)
def filamentarity(width, length):
    """
    Calculate the filamentarity of an object.

    Parameters:
    width (float): The width of the object.
    length (float): The length of the object.

    Returns:
    float: The filamentarity value of the object.
    """

    f = (length - width) / (length + width)
    return f

@jit(nopython=True)
def calculate_distance_and_set_position(i, j, k, x, y, z, r, volume):
    """
    Calculate the distance between a point and the center of the sphere and set the position in the volume.

    Parameters:
    - i (int): The x-coordinate of the point.
    - j (int): The y-coordinate of the point.
    - k (int): The z-coordinate of the point.
    - x (int): The x-coordinate of the center of the sphere.
    - y (int): The y-coordinate of the center of the sphere.
    - z (int): The z-coordinate of the center of the sphere.
    - r (int): The radius of the sphere.
    - volume (ndarray): A 3D numpy array with dimensions (grid_dimensions, grid_dimensions, grid_dimensions).
                       The array represents a volume where everything is marked as False except for the positions
                       within the ball, which are marked as True.
    """

    distance = np.sqrt((i - x) ** 2 + (j - y) ** 2 + (k - z) ** 2)
    if distance <= r:
        volume[i, j, k] = True



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
    # Create a boolean numpy array to represent the volume
    volume = np.zeros((grid_dimensions, grid_dimensions, grid_dimensions), dtype=bool)
    
    # Iterate over each position in the volume
    for i in tqdm(range(grid_dimensions)):
        for j in range(grid_dimensions):
            for k in range(grid_dimensions):
                calculate_distance_and_set_position(i, j, k, x, y, z, r, volume)
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

    m0, m1, m2, m3 = minkowski_functionals
    v0 = m0 
    v1 = m1 * 4 / 3
    v2 = m2 * 2 * np.pi / 3
    v3 = m3 * 4 * np.pi / 3
    
    return v0, v1, v2, v3

grid_dimensions = 300 #Number of grid points in each dimension
center = [grid_dimensions//2, grid_dimensions//2, grid_dimensions//2] #Center of the grid

radius = 75 #Radius of the sphere

minkowski_functionals = compute_minkowski_functionals(radius)

print("V_0: ", minkowski_functionals[0])
print("V_1: ", minkowski_functionals[1])
print("V_2: ", minkowski_functionals[2])
print("V_3: ", minkowski_functionals[3])


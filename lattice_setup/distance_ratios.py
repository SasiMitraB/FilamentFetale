import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import jit
from quantimpy import minkowski as mk


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
def calculate_distance_and_set_position(i, j, k, x, y, z, r, grid):
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
        i = i % grid_dimensions
        j = j % grid_dimensions
        k = k % grid_dimensions
        grid[i, j, k] = True
    


def add_balls(center, radius, grid):
    """
    Add balls to the grid around the specified center with the given radius.
    Args:
        center (tuple): The coordinates of the center of the balls (x, y, z).
        radius (int): The radius of the balls.
        grid (list): The grid to add the balls to.
    Returns:
        list: The updated grid with the added balls.
    """
    x_center, y_center, z_center = center

    x_min = int(x_center - ((radius + 3)//1))
    x_max = int(x_center + ((radius + 3)//1))
    y_min = int(y_center - ((radius + 3)//1))
    y_max = int(y_center + ((radius + 3)//1))
    z_min = int(z_center - ((radius + 3)//1))
    z_max = int(z_center + ((radius + 3)//1))


    for i in tqdm(range(x_min, x_max)):
        for j in range(y_min, y_max):
            for k in range(z_min, z_max):
                calculate_distance_and_set_position(i, j, k, x_center, y_center, z_center, radius, grid)

    return grid




def add_balls_old(center, radius, grid):
    """
    Takes input of the center and radius, adds a ball at the specified center and radius, and returns the volume.

    Parameters:
    - center (tuple): The coordinates of the center of the ball in the form (x, y, z).
    - radius (float): The radius of the ball.

    Returns:
    - grid (ndarray): A 3D numpy array with dimensions (grid_dimensions, grid_dimensions, grid_dimensions).
                       The array represents a volume where everything is marked as False except for the positions
                       within the ball, which are marked as True.

    """
    x, y, z = center
    r = radius
    
    # Iterate over each position in the volume
    for i in tqdm(range(grid_dimensions)):
        for j in range(grid_dimensions):
            for k in range(grid_dimensions):
                calculate_distance_and_set_position(i, j, k, x, y, z, r, grid)
    return grid

def plot_volume(volume):
    """
    Plot the given volume object in a 3D scatter plot.
    Parameters:
    volume (ndarray): A 3D numpy array representing the volume.
    Returns:
    None
    """
    # Plot the volume object in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Get the indices of the True values in the volume
    indices = np.where(volume)

    # Plot the True values as scatter points
    ax.scatter(indices[0], indices[1], indices[2], c='b', marker='o')

    # Set the plot limits
    ax.set_xlim(0, grid_dimensions)
    ax.set_ylim(0, grid_dimensions)
    ax.set_zlim(0, grid_dimensions)

    # Set the plot labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
    
def compute_minkowski_functionals(ball_radius):
    """
    Create a volume with balls and compute the Minkowski functionals.

    Parameters:
    - ball_radius: Radius of the balls to be added

    Returns:
    - Minkowski functionals computed for the volume
    """
    
    volume = np.zeros((grid_dimensions, grid_dimensions, grid_dimensions), dtype=bool)

    # Add balls to the volume
    for center in list_of_centers:
        volume = add_balls(center, ball_radius, volume)
    
    # Compute the Minkowski functionals for the volume
    minkowski_functionals = mk.functionals(volume)

    m0, m1, m2, m3 = minkowski_functionals

    v0 = m0 * (unit_grid_dimension**3)  #Unit Grid Dimension raised to 3
    v1 = (m1 * 4 / 3) * (unit_grid_dimension**2) #Unit Grid Dimension raised to 2
    v2 = (m2 * 2 * np.pi / 3) * unit_grid_dimension #Unit Grid Dimension raised to 1
    v3 = m3 * 4 * np.pi / 3 #Dimensionless Constant
    
    return v0, v1, v2, v3

# Define the grid dimensions and the radius of the sphere
grid_dimensions = 500 #No of grid boxes in each dimensions
length_of_grid = 300 #Length of the grid in meters
radius = 35 #Radius of the sphere in meters

number_of_balls = 1 #Number of balls to be placed in the grid



unit_grid_dimension = length_of_grid / grid_dimensions

radius_in_grid = radius / unit_grid_dimension
#Randomly Generating number_of_balls centers to place the balls
#list_of_centers = [[np.random.randint(2, grid_dimensions - 10), np.random.randint(2, grid_dimensions - 10), np.random.randint(2, grid_dimensions - 10)] for i in range(number_of_balls)]
list_of_centers = [[grid_dimensions//2, grid_dimensions//2, grid_dimensions//2]]

minkowski_functionals = compute_minkowski_functionals(radius_in_grid)

print("V_0: ", minkowski_functionals[0])
print("V_1: ", minkowski_functionals[1])
print("V_2: ", minkowski_functionals[2])
print("V_3: ", minkowski_functionals[3])


print("Shapefinders: ", calculate_shapefinders(minkowski_functionals))
thickness, width, length = calculate_shapefinders(minkowski_functionals)

print("Planarity: ", planarity(thickness, width))
print("Filamentarity: ", filamentarity(width, length))
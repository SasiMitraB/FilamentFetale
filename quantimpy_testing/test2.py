import numpy as np
from numba import jit

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
                i = i % grid_dimensions
                j = j % grid_dimensions
                k = k % grid_dimensions
                calculate_distance_and_set_position(i, j, k, x_center, y_center, z_center, radius, grid)

    return grid



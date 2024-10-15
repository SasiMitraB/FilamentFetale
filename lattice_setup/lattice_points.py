import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import requests

def send_discord_notification(webhook_url, message, file_path=None):
    payload = {
        'content': message
    }
    
    # Prepare the request
    if file_path:
        # If there is an image file to upload
        with open(file_path, 'rb') as f:
            files = {
                'file': (file_path, f)
            }
            response = requests.post(webhook_url, data=payload, files=files)
    else:
        # If there is no image file to upload
        response = requests.post(webhook_url, json=payload)

    # Handle response
    if response.status_code in [200, 204]:
        print("Notification sent successfully!")
    else:
        print(f"Failed to send notification. Status code: {response.status_code}")
 

# Replace with your Discord webhook URL
webhook_url = 'https://discord.com/api/webhooks/1275399280340897807/hRdISUa2AYdqB1jYJvClv6ENTgJWdgVSl-bHcdJ_-_i5zN56YyODlkBuD3GOPNwCdKaA'


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
    

@jit(nopython = True)
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


    for i in range(x_min, x_max):
        for j in range(y_min, y_max):
            for k in range(z_min, z_max):
                calculate_distance_and_set_position(i, j, k, x_center, y_center, z_center, radius, grid)

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
    webhook_url = 'https://discord.com/api/webhooks/1285587084073107517/aYMnqG6LogHVeuQf06esyKKGjUkZUvQp37Aej7pexoLz-JIrSUMJYNntyj0KPxhLx4an'
    plt.savefig("spam_plot.png")
    send_discord_notification(webhook_url=webhook_url,
                              message="Spam Plot of the Volume",
                              file_path='spam_plot.png')

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
    
    plot_volume(volume)
    
    return 1,2,3,4

# Define the grid dimensions and the radius of the sphere
grid_dimensions = 500 #No of grid boxes in each dimensions
length_of_grid = 1 #Length of the grid in meters

unit_grid_dimension = length_of_grid / grid_dimensions


#Generating a Lattice and Placing the balls in the Centers of the lattices
lattice_dimensions = 3  # Number of lattice points per dimension
# Calculate spacings
spacing_lattice = length_of_grid / lattice_dimensions
spacing_grid = length_of_grid / grid_dimensions

# Generate lattice centers
list_of_centers_lattice = []
for i in range(lattice_dimensions):
    for j in range(lattice_dimensions):
        for k in range(lattice_dimensions):
            center_lattice = [i * spacing_lattice + spacing_lattice / 2,
                              j * spacing_lattice + spacing_lattice / 2,
                              k * spacing_lattice + spacing_lattice / 2]
            list_of_centers_lattice.append(center_lattice)

# Convert lattice centers to grid coordinates
list_of_centers_grid = []
for center in list_of_centers_lattice:
    center_grid = [int(center[0] / spacing_grid),
                   int(center[1] / spacing_grid),
                   int(center[2] / spacing_grid)]
    list_of_centers_grid.append(center_grid)

list_of_centers = list_of_centers_grid

# Verify the number of generated centers
number_of_balls = len(list_of_centers_grid)
print(f"Number of balls: {number_of_balls}")

print('The length of the grid is', length_of_grid, 'meters')
print('The Grid Resolution is', grid_dimensions)
print('There are', number_of_balls, 'balls placed in a lattice in the volume')

radius = 0.1 #Radius of the sphere in meters

radius_in_grid = radius / unit_grid_dimension

minkowski_functionals = compute_minkowski_functionals(radius_in_grid)
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import jit
from quantimpy import minkowski as mk
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
length_of_grid = 1 #Length of the grid in meters

number_of_balls = 100  #Number of balls to be placed in the grid

unit_grid_dimension = length_of_grid / grid_dimensions
#radius_in_grid = radius / unit_grid_dimension

# Number of simulations
n_simulations = 10

print('The length of the grid is', length_of_grid, 'meters')
print('The Grid Resolution is', grid_dimensions)
print('There are', number_of_balls, 'balls placed randomly in the volume')
print("The number of simulations are:", n_simulations)

# Number of radii
n_radii = 50
list_of_radii = np.asarray([0.05 + i * 0.01 for i in range(n_radii)])
list_of_radii_in_grid = list_of_radii / unit_grid_dimension

# Pre-allocate a numpy array to hold the Minkowski functionals for all runs
all_minkowski_functionals = np.zeros((n_simulations, n_radii, 4))

# Loop over the number of simulations
for sim_idx in range(n_simulations):
    # Randomly generate centers for the balls
    list_of_centers = np.array([[100, 
                                 np.random.randint(2, grid_dimensions - 10), 
                                 np.random.randint(2, grid_dimensions - 10)] 
                                 for i in range(number_of_balls)])
    print("Iteration Number", sim_idx)
    # Compute the Minkowski functionals for each radius in this simulation
    for radius_idx, radius_in_grid in enumerate(tqdm(list_of_radii_in_grid)):
        v0, v1, v2, v3 = compute_minkowski_functionals(radius_in_grid)
        all_minkowski_functionals[sim_idx, radius_idx, :] = [v0, v1, v2, v3]

# Compute the mean and standard deviation along the simulations axis (axis=0) for each Minkowski functional
mean_minkowski = np.mean(all_minkowski_functionals, axis=0)  # Shape will be (n_radii, 4)
std_minkowski = np.std(all_minkowski_functionals, axis=0)    # Shape will be (n_radii, 4)


fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Plot for V0
axs[0, 0].plot(list_of_radii, mean_minkowski[:, 0], "o-", label='Mean V0')
axs[0, 0].fill_between(list_of_radii, 
                       mean_minkowski[:, 0] - std_minkowski[:, 0], 
                       mean_minkowski[:, 0] + std_minkowski[:, 0], 
                       color='gray', alpha=0.3, label='±1 Std Dev')
axs[0, 0].set_title('V0')
axs[0, 0].set_xlabel('Radius')
axs[0, 0].set_ylabel('Minkowski Functional')
axs[0, 0].legend()

# Plot for V1
axs[0, 1].plot(list_of_radii, mean_minkowski[:, 1], "o-", label='Mean V1')
axs[0, 1].fill_between(list_of_radii, 
                       mean_minkowski[:, 1] - std_minkowski[:, 1], 
                       mean_minkowski[:, 1] + std_minkowski[:, 1], 
                       color='gray', alpha=0.3, label='±1 Std Dev')
axs[0, 1].set_title('V1')
axs[0, 1].set_xlabel('Radius')
axs[0, 1].set_ylabel('Minkowski Functional')
axs[0, 1].legend()

# Plot for V2
axs[1, 0].plot(list_of_radii, mean_minkowski[:, 2], "o-", label='Mean V2')
axs[1, 0].fill_between(list_of_radii, 
                       mean_minkowski[:, 2] - std_minkowski[:, 2], 
                       mean_minkowski[:, 2] + std_minkowski[:, 2], 
                       color='gray', alpha=0.3, label='±1 Std Dev')
axs[1, 0].set_title('V2')
axs[1, 0].set_xlabel('Radius')
axs[1, 0].set_ylabel('Minkowski Functional')
axs[1, 0].legend()

# Plot for V3
axs[1, 1].plot(list_of_radii, mean_minkowski[:, 3], "o-", label='Mean V3')
axs[1, 1].fill_between(list_of_radii, 
                       mean_minkowski[:, 3] - std_minkowski[:, 3], 
                       mean_minkowski[:, 3] + std_minkowski[:, 3], 
                       color='gray', alpha=0.3, label='±1 Std Dev')
axs[1, 1].set_title('V3')
axs[1, 1].set_xlabel('Radius')
axs[1, 1].set_ylabel('Minkowski Functional')
axs[1, 1].legend()

plt.tight_layout()

# Save the figure
plt.savefig('random_process.png')



# Notify Discord with the message and image file

message = """
Random Placement, Mean and Variance
Length of Grid: {length_of_grid} m
Grid Resolution: {grid_dimensions} per dimension
Unit Grid Box Size: {unit_grid_dimension} m
Number of Balls Placed: {number_of_balls}
""".format(length_of_grid=length_of_grid, grid_dimensions=grid_dimensions, 
           unit_grid_dimension=unit_grid_dimension, number_of_balls=number_of_balls)


send_discord_notification(webhook_url, message, "random_process.png")

plt.show()

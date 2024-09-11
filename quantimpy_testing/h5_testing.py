import numpy as np
import h5py


grid_dimensions = 12 #Number of grid points in each dimension
center = [grid_dimensions//2, grid_dimensions//2, grid_dimensions//2] #Center of the grid

radius = 6 #Radius of the sphere

volume_object = np.zeros((grid_dimensions, grid_dimensions, grid_dimensions))
x, y, z = center
r = radius
# Create a grid of coordinates
coords = np.indices((grid_dimensions, grid_dimensions, grid_dimensions))

# Calculate the squared distance from the center (x, y, z)
squared_distances = (coords[0] - x) ** 2 + (coords[1] - y) ** 2 + (coords[2] - z) ** 2

# Create a mask where the squared distance is less than or equal to r^2
mask = squared_distances < r ** 2

# Apply the mask to the volume array
volume_object[mask] = 1

# Create a new HDF5 file
with h5py.File('volume_object.h5', 'w') as f:
    # Create a dataset to store the volume_object
    f.create_dataset('volume_object', data=volume_object)


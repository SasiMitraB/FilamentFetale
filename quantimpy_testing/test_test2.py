import unittest
import numpy as np
from tqdm import tqdm
from test2 import calculate_distance_and_set_position, add_balls

class TestQuantimpyFunctions(unittest.TestCase):

    def test_calculate_distance_and_set_position_inside_sphere(self):
        grid_dimensions = 10
        grid = np.zeros((grid_dimensions, grid_dimensions, grid_dimensions), dtype=bool)
        calculate_distance_and_set_position(5, 5, 5, 5, 5, 5, 3, grid)
        self.assertTrue(grid[5, 5, 5])

    def test_calculate_distance_and_set_position_outside_sphere(self):
        grid_dimensions = 10
        grid = np.zeros((grid_dimensions, grid_dimensions, grid_dimensions), dtype=bool)
        calculate_distance_and_set_position(9, 9, 9, 5, 5, 5, 3, grid)
        self.assertFalse(grid[9, 9, 9])

    def test_add_balls(self):
        grid_dimensions = 10
        grid = np.zeros((grid_dimensions, grid_dimensions, grid_dimensions), dtype=bool)
        center = (5, 5, 5)
        radius = 3
        updated_grid = add_balls(center, radius, grid)
        self.assertTrue(np.any(updated_grid))

if __name__ == '__main__':
    unittest.main()
import numpy as np


def generate_grids(
    n_dims: int,
    num_points: int,
    upper_bound: np.ndarray | list,
    lower_bound: np.ndarray | list,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates an evenly distributed grid across n dimensions and its midpoints.
    By default, the grid goes from 0-1

    @param n_dims: number of dimensions
    @param num_points: number of points per dimension (same in all dimensions)
    @param upper_bound: if provided the grid is rescaled
    @param lower_bound:
    @return:
    """
    upper_bound = np.array(upper_bound)
    lower_bound = np.array(lower_bound)
    # Create an evenly spaced grid in each dimension
    points = np.linspace(0, 1, num_points)

    # Generate the full grid by creating a meshgrid and then stacking the result
    grids = np.meshgrid(*([points] * n_dims), indexing="ij")
    grid = np.stack(grids, axis=-1).reshape(-1, n_dims)

    # Calculate midpoints by averaging adjacent points in the grid
    midpoints = (points[:-1] + points[1:]) / 2
    midpoints_grids = np.meshgrid(*([midpoints] * n_dims), indexing="ij")
    midpoints_grid = np.stack(midpoints_grids, axis=-1).reshape(-1, n_dims)

    grid = lower_bound + grid * (upper_bound - lower_bound)
    midpoints_grid = lower_bound + midpoints_grid * (upper_bound - lower_bound)

    return grid, midpoints_grid

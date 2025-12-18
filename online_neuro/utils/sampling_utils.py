# online_neuro/utils/sampling_utils.py
import numpy as np


def generate_grids(
    n_dims: int,
    num_points: int,
    upper_bound: np.ndarray | list,
    lower_bound: np.ndarray | list,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate two numpy arrays representing an evenly distributed grid across
    `n_dims` dimensions and the center points (midpoints) of the resulting grid cells.

    The grid is initially created in the unit hypercube [0, 1]^D and is then
    rescaled to the physical bounds defined by `lower_bound` and `upper_bound`.

    Parameters
    ----------
    n_dims : int
        The number of dimensions (features) in the space.
    num_points : int
        The number of evenly spaced points to create along each dimension.
    upper_bound : ArrayLike (np.ndarray or list of floats/ints)
        The maximum value for each dimension in the physical space (shape: (n_dims,)).
    lower_bound : ArrayLike (np.ndarray or list of floats/ints)
        The minimum value for each dimension in the physical space (shape: (n_dims,)).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing:
        1. `grid`: A 2D array of all points on the regular grid (shape: (num_points^n_dims, n_dims)).
        2. `midpoints_grid`: A 2D array of the centers of all cells defined by the grid
           (shape: ((num_points-1)^n_dims, n_dims)).
    """
    upper_bound = np.array(upper_bound, dtype=float)
    lower_bound = np.array(lower_bound, dtype=float)

    if len(upper_bound) != n_dims or len(lower_bound) != n_dims:
        raise ValueError(
            f"Bounds must match n_dims ({n_dims}). Got bounds of length {len(upper_bound)}."
        )

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

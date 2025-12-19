# simulators/python/problems/toy_problems.py
from typing import Any, Iterable

import numpy as np


def circle(
    x0: Iterable,
    x1: Iterable,
    radius: Iterable | float = 0.5,
    noise: Iterable | float = 0.0,
    center: Iterable = np.array([0, 0]),
    *args: Any,
    **kwargs: Any,
):
    """
    Toy ptoblem.
    Determines whether 2D coordinates are inside or outside a defined circle.

    The decision is based on comparing the calculated distance from the center
    to the defined radius.
    Optionally, Gaussian noise can be added to the measurements.

    Parameters
    ----------
    x0 : array_like
        The first coordinate (x-axis) of the points. Must be broadcastable
        with `x1`.
    x1 : array_like
        The second coordinate (y-axis) of the points. Must be broadcastable
        with `x0`.
    radius : array_like or float, optional
        The radius of the circle. Can be a scalar or an array if a varying
        radius comparison is desired for multiple points. Default is 0.5.
    noise : array_like or float, optional
        The standard deviation (sigma) of the Gaussian noise applied to the
        calculated distance before comparison. Noise is added randomly
        every time the function is called. Default is 0.0.
    center : array_like, optional
        The [x, y] coordinates for the circle center. Must be a 2-element
        array. Default is [0.0, 0.0].
    *args : Any
        Positional arguments received for compatibility (ignored).
    **kwargs : Any
        Keyword arguments received for compatibility (ignored).

    Returns
    -------
    obs : dict
        A dictionary containing the classification results under the key
        "observations". The list contains integers:
        1 if sample inside the circle, 0 if the point is outside the circle.

    Notes
    -----
    The noise is applied to the calculated Euclidean distance, simulating
    measurement error in the distance itself, rather than perturbing the
    input coordinates.
    """
    # Ensure  parameters are NumPy arrays
    x0 = np.asarray(x0)
    x1 = np.asarray(x1)

    center = np.asarray(center)
    radius_arr = np.asarray(radius)
    noise_magnitude = np.asarray(noise)

    # Use broadcasting to handle scalars or arrays and stack them into (N, 2)
    try:
        x = np.stack([x0, x1], axis=-1)
    except ValueError as e:
        raise ValueError(f"Input arrays x0 and x1 must be broadcastable: {e}")

    if center.shape != (2,):
        raise ValueError("Center must be a 2-element array [x, y].")

    # 1. Calculate Euclidean distance from center
    x_centered = x - center
    radii = np.linalg.norm(x_centered, axis=-1)

    # 2. Apply noise if required
    if np.any(noise_magnitude > 0):
        # Generate standard Gaussian noise for each point
        standard_noise = np.random.randn(*radii.shape)

        # Scale noise by the provided magnitude (handles broadcasting of noise)
        radii = radii + noise_magnitude * standard_noise

    # 3. Comparison
    # Return 1 if distance (radii) is greater than the radius, else 0.
    result = (radii > radius_arr).astype(int)

    # Convert to list for the required output format
    obs = {"observations": result.tolist()}
    return obs


def multiple_circles(
    x0: Iterable[float],
    x1: Iterable[float],
    radius: Iterable[float | Iterable[float]],
    noise: Iterable[float | Iterable[float]],
    center: Iterable[Iterable[float]],
    *args: Any,
    **kwargs: Any,
) -> dict[str, list]:
    """
    Toy problem
    Evaluates 2D coordinates against a set of multiple circles defined by
    varying radii, noise levels, and centers.

    The final classification returns 1 only if the point is determined to be
    outside ALL defined circles.

    Parameters
    ----------
    x0 : Iterable of float
        The first coordinate (x-axis) for all points to be evaluated.
    x1 : Iterable of float
        The second coordinate (y-axis) for all points to be evaluated.
    radius : Iterable of (float or Iterable of float)
        A sequence defining the radius parameters for each circle. Each element
        in the sequence corresponds to one circle's radius (which can be a
        scalar or an array if broadcastable with the number of points).
    noise : Iterable of (float or Iterable of float)
        A sequence defining the Gaussian noise standard deviations for each
        circle. Each element corresponds to one circle's noise level.
    center : Iterable of Iterable of float
        A sequence defining the centers for each circle. Each element must be
        a 2-element array or list [x, y] defining the center coordinates.
    *args : Any
        Positional arguments received for compatibility (ignored).
    **kwargs : Any
        Keyword arguments received for compatibility (ignored).

    Returns
    -------
    obs : dict
        A dictionary containing the combined binary results under the key
        "observations". The list contains integers:
        1 if sample inside the circle, 0 if the point is outside the circle.

    Raises
    ------
    ValueError
        If the lengths of `radius`, `noise`, and `center` sequences do not match.

    """

    results = []

    for r, n, c in zip(radius, noise, center):
        result = circle(x0, x1, radius=r, noise=n, center=c)
        results.append(result["observations"])

    # Sum the results from all circles and return binary results
    combined_results = np.sum(results, axis=0) - (len(radius) - 1)
    binary_results = (combined_results > 0).astype(int)
    binary_results = binary_results.tolist()
    obs = {"observations": binary_results}

    return obs


def hypersphere(
    points: Iterable[Iterable[float]],
    radius: float | Iterable[float],
    noise: float | Iterable[float] = 0.0,
    center: Iterable[float] | None = None,
    *args: Any,
    **kwargs: Any,
) -> dict[str, list]:
    """
    Determines whether N-dimensional coordinates are inside or outside a defined hypersphere.

    The decision is based on comparing the calculated distance from the center
    to the defined radius, potentially introducing uniform noise to the radius.

    Parameters
    ----------
    points : array_like, shape (N_points, N_dimensions)
        The coordinates of the points to be evaluated.
    radius : float or array_like
        The radius of the hypersphere.
    noise : float or array_like, optional
        The half-width (maximum deviation) of the uniform noise applied to the
        radius, such that the noisy radius is calculated as
        R_noisy = R + U(-noise, noise). Default is 0.0 (no noise).
    center : array_like, shape (N_dimensions,), optional
        The coordinates for the hypersphere center. If None, assumes center
        is at the origin (0, 0, ...). Default is None.
    *args : Any
        Positional arguments received for compatibility (ignored).
    **kwargs : Any
        Keyword arguments received for compatibility (ignored).

    Returns
    -------
    obs : dict
        A dictionary containing the classification results under the key
        "observations". The list contains integers:
        1 if sample inside the sphere, 0 if the point is outside the sphere.

    Notes
    -----
    The noise is applied to the *radius* itself using a uniform distribution.
    """
    points = np.asarray(points)

    if center is None:
        # Default to origin based on the dimension of the input points
        center = np.zeros(points.shape[1])
    else:
        center = np.asarray(center)

    radius = np.asarray(radius)
    noise = np.asarray(noise)

    # Calculate Euclidean distances from the center
    distances = np.linalg.norm(points - center, axis=1)

    # Apply uniform noise to the radius: R_noisy = R + U(-N, N)
    if np.any(noise > 0):
        # Generate uniform random values in [-1, 1] scaled by noise magnitude
        noise_vector = noise * np.random.uniform(
            low=-1.0, high=1.0, size=distances.shape
        )

        # Ensure radius is correctly broadcastable if it's a scalar
        if radius.ndim == 0:
            radius = np.full(distances.shape, radius)

        noisy_radius = radius + noise_vector
    else:
        noisy_radius = radius

    # Result: 1 if INSIDE (distance <= noisy_radius), 0 if OUTSIDE
    result = (distances <= noisy_radius).astype(int)

    obs = {"observations": result.tolist()}
    return obs


def multiple_hyperspheres(
    points: Iterable[Iterable[float]],
    radii: list[float | Iterable[float]],
    noises: list[float | Iterable[float]],
    centers: list[Iterable[float]],
    *args: Any,
    **kwargs: Any,
) -> dict[str, list]:
    """
    Evaluates N-dimensional coordinates against a set of multiple hyperspheres.

    The classification returns 1 only if the point is determined to be
    inside ALL defined hyperspheres (logical AND operation on the 'inside' state).

    Parameters
    ----------
    points : array_like, shape (N_points, N_dimensions)
        The coordinates of the points to be evaluated.
    radii : List of (float or Iterable of float)
        A list defining the radius parameters for each hypersphere.
    noises : List of (float or Iterable of float)
        A list defining the uniform noise half-widths for each hypersphere.
    centers : List of Iterable of float
        A list defining the centers for each hypersphere. Each element must be
        a sequence defining the center coordinates (shape N_dimensions,).
    *args : Any
        Positional arguments received for compatibility (ignored).
    **kwargs : Any
        Keyword arguments received for compatibility (ignored).

    Returns
    -------
    obs : dict
        A dictionary containing the combined binary results under the key
        "observations". The list contains integers:
        1 if sample inside any of the hyperspheres else 0.

    Raises
    ------
    ValueError
        If the lengths of `radii`, `noises`, and `centers` lists do not match.

    """
    N_spheres = len(radii)
    if not (N_spheres == len(noises) == len(centers)):
        raise ValueError(
            "The lists for radii, noises, and centers must all have the same length."
        )

    results = []

    # Iterate and evaluate against each hypersphere
    for r, n, c in zip(radii, noises, centers):
        # We rely on the `hypersphere` function
        result = hypersphere(points, radius=r, noise=n, center=c)
        results.append(result["observations"])

    results_array = np.asarray(results)

    # Combination Logic (Logical AND: Inside any hyperspheres)
    # Since hypersphere returns 1 for INSIDE, we sum the results.
    # If sum == N_spheres, the point is inside every sphere.
    combined_results = np.sum(results_array, axis=0).astype(int)

    # 4. Output formatting
    obs = {"observations": combined_results.tolist()}
    return obs


def log_single_var(
    x: np.ndarray, noise: float = 0, target_loc: float = 1, *args: Any, **kwargs: Any
) -> dict[str, list[int]]:
    """
    A one-input logistic regression problem with noise.
    Noise is sampled each call.

    Parameters
    ----------
    x : np.ndarray
        Input feature array.
    noise : float, optional
        Noise level to add to `x` (default is 0). Noise is sampled from a normal distribution.
    target_loc : float, optional
        Threshold for classification (default is 1).
    *args
        Additional positional arguments (not used).
    **kwargs
        Additional keyword arguments (not used).

    Returns
    -------
    dict
        Dictionary containing:
        - 'observations' : list of int
            Binary output (0 or 1) for each input in `x`.
    """
    if noise > 0:
        x = x + noise * np.random.normal(size=len(x))
    result = np.where(x > target_loc, 1, 0).tolist()
    return {"observations": result}


def toy_feasbility(
    x: float, ymax: float = 1, noise: float = 0, *args: Any, **kwargs: Any
) -> dict[str, list[float]]:
    """
    Toy feasibility function with noise.

    Parameters
    ----------
    x : float
        Input value.
    ymax : float, optional
        Maximum reference value (default is 1).
    noise : float, optional
        Standard deviation of Gaussian noise added to the output (default is 0).
    *args
        Additional positional arguments (not used).
    **kwargs
        Additional keyword arguments (not used).

    Returns
    -------
    dict
        Dictionary containing:
        - 'observations' : list of float
            Computed scalar output as a list.
    """
    term1 = -0.001 / (
        0.01 * ((ymax - 5 / (4 * np.pi**2) * x**2 + (5 / np.pi) * x - 2) ** 2)
    )
    term2 = 0.04 * (1 - 1 / (5 * np.pi)) * np.cos(x) * np.cos(ymax)
    term3 = 0.05 * np.log(x**2 + ymax**2 + 1)
    result = term1 + term2 + term3 + 1 + 0.3
    if noise > 0:
        result += np.random.normal(loc=0.0, scale=noise)
    return {"observations": [result]}


def vlmop2(x: np.ndarray, *args: Any, **kwargs: Any) -> dict[str, list[list[float]]]:
    """
    VLMOP2 multi-objective test function.

    Parameters
    ----------
    x : np.ndarray
        Input array with shape (n_samples, 2).
    *args
        Additional positional arguments (not used).
    **kwargs
        Additional keyword arguments (not used).

    Returns
    -------
    dict
        Dictionary containing:
        - 'observations' : list of list of float
            Two lists corresponding to the two objectives for each input sample.
    """
    transl = 1 / np.sqrt(2)
    part1 = (x[:, 0] - transl) ** 2 + (x[:, 1] - transl) ** 2
    part2 = (x[:, 0] + transl) ** 2 + (x[:, 1] + transl) ** 2

    y0 = 1 - np.exp(-part1)
    y1 = 1 - np.exp(-part2)

    return {"observations": [y0.tolist(), y1.tolist()]}


def rosenbrock(
    x0: float, x1: float, a: float = 1.0, b: float = 100.0, *args: Any, **kwargs: Any
) -> dict[str, list[float]]:
    """
    Rosenbrock function (2D).

    Parameters
    ----------
    x0 : float
        First input coordinate.
    x1 : float
        Second input coordinate.
    a : float, optional
        Rosenbrock parameter 'a' (default 1.0).
    b : float, optional
        Rosenbrock parameter 'b' (default 100.0).
    *args
        Additional positional arguments (not used).
    **kwargs
        Additional keyword arguments (not used).

    Returns
    -------
    dict
        Dictionary containing:
        - 'observations' : list of float
            Rosenbrock function value as a single-element list.
    """
    value = (a - x0) ** 2 + b * (x1 - x0**2) ** 2
    return {"observations": [value]}

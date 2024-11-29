import tensorflow as tf
import numpy as np
# Closely equivalent problems to the ones in Matlab, but here the python versions of them.


def circle(x0, x1, radius=0.5, noise=0, center=np.array([0, 0])):
    """
    Given a center, radius and noise, define whether a given coordinate (x0, x1) is in or out of a circle.

    Noise is at random every time the function is called
    Noise is in the "distance" measurement (not the coordinates).
    @param x0: First coordinate
    @param x1: Second coordinate
    @param radius:
    @param noise: Noise level (Normal random)
    @param center: [x, y] coordinates for the circle center.
    @return:
    """
    x0 = np.atleast_2d(np.array(x0)).reshape(-1,1)
    x1 = np.atleast_2d(np.array(x1)).reshape(-1,1)

    x = np.hstack([x0, x1])

    # Calculate distances from the center for all points
    x_centered = x - center
    radii = np.linalg.norm(x_centered, axis=1)

    # Add noise to the distance if required
    if noise > 0:
        noise_vector = noise * np.random.randn(len(radii))
        radii = radii + noise_vector

    # Return 1 if the point is outside the radius, otherwise return 0 (opposite of original Python function)
    return (radii > radius).astype(int)

def multiple_circles(x0, x1, radius, noise, center):
    """
    Parameters:
    x (np.ndarray): Points to be evaluated, shape (n_points, 2).
    radius (list of float): List of radii for the circles.
    noise (list of float): List of noise values for each circle.
    center (list of np.ndarray): List of [x, y] center coordinates for each circle.
        
    Returns:
    np.ndarray: Combined binary results for all circles, shape (n_points,).
    """
    results = []

    for r, n, c in zip(radius, noise, center):
        result = circle(x0, x1, radius=r, noise=n, center=c)
        results.append(result)
    
    # Sum the results from all circles and return binary results
    combined_results = np.sum(results, axis=0) - (len(radius) - 1)
    binary_results = (combined_results > 0).astype(int)

    return binary_results

def log_single_var(x, noise=0, target_loc=1):
    """
    A one input logistic regression problem with noise.
    Noise is sampled each call.

    @param x: Single feature
    @param noise: Noise level (Normal random)
    @param target_loc: The real threshold.
    @return:
    """
    if noise > 0:
        x += 0.2 * noise * np.random.normal(size=len(x))
    return tf.cast(tf.where(x > target_loc, 1, 0), tf.float64)


def toy_feasbility(x, ymax=1, noise=0):
    """
    TODO extend it to have noise

    @param x:
    @param ymax:
    @param noise:
    @return:
    """
    term1 = -0.001 / (0.01 * ((ymax - 5 / (4 * np.pi ** 2) * x ** 2 + (5 / np.pi) * x - 2) ** 2))
    term2 = 0.04 * (1 - 1 / (5 * np.pi)) * np.cos(x) * np.cos(ymax)
    term3 = 0.05 * np.log(x ** 2 + ymax ** 2 + 1)
    return term1 + term2 + term3 + 1 + 0.3


def vlmop2(x):
    transl = 1/np.sqrt(2)
    # Compute part1 and part2
    part1 = (x[:, 0] - transl) ** 2 + (x[:, 1] - transl) ** 2
    part2 = (x[:, 0] + transl) ** 2 + (x[:, 1] + transl) ** 2

    # Calculate y0 and y1
    y0 = 1 - np.exp(-part1)
    y1 = 1 - np.exp(-part2)

    return [y0, y1]


def rosenbrock(x):
    #TODO implement
    return NotImplementedError

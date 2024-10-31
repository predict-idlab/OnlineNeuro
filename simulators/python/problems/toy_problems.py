import tensorflow as tf
import numpy as np
# Closely equivalent problems to the ones in Matlab, but here the python versions of them.
# Can be handy for writing code in Notebooks.


def circle(x, radius=0.5, noise=0, center=[0, 0]):
    x_centered = x - center
    radii = tf.sqrt(tf.reduce_sum(tf.square(x_centered), axis=1, keepdims=True))
    if noise > 0:
        return tf.cast((radii - radius) > 0, tf.float64)

    else:
        return tf.cast((radii - radius + noise * np.random.normal(size=radii.shape) > 0), tf.float64)


def log_single_var(x, noise=0, target_loc=1):
    if noise > 0:
        x += 0.2 * noise * np.random.norma(size=len(x))
    return tf.cast(tf.where(x > target_loc, 1, 0), tf.float64)


def toy_feasbility(x, ymax=1, noise=0):
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

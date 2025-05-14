# simulators/python/problems/__init__.py
import warnings
from simulators.python.problems.toy_problems import (
    circle,
    multiple_circles,
    hypersphere,
    multiple_hyperspheres,
    log_single_var,
    toy_feasbility,
    vlmop2,
    rosenbrock

)
try:
    from simulators.python.problems import cajal_problems
except ImportError as e:
    warnings.warn('Cajal is not installed, Cajal problems are not loaded.')

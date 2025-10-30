# simulators/python/problems/__init__.py
import warnings

from common.utils import configure_python_paths
from simulators.python.problems.toy_problems import (
    circle,
    hypersphere,
    log_single_var,
    multiple_circles,
    multiple_hyperspheres,
    rosenbrock,
    toy_feasbility,
    vlmop2,
)

try:
    import cajal
except ImportError:
    try:
        warnings.warn("Cajal is not installed, trying to load from path in config.json")
        configure_python_paths()
        import cajal
    except Exception as e:
        raise ImportError(f"cajal is not installed. Install it to use this module: {e}")

__all__ = [
    "circle",
    "hypersphere",
    "log_single_var",
    "multiple_circles",
    "multiple_hyperspheres",
    "rosenbrock",
    "toy_feasbility",
    "vlmop2",
    "cajal",
]

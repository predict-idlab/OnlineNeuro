from functools import wraps
from typing import Any, Callable


def ensure_lists(func: Callable) -> Callable:
    """
    Decorator to ensure that all arguments passed to a function are lists.

    Positional and keyword arguments that are not lists will be wrapped
    into a single-element list before calling the original function.

    Parameters
    ----------
    func : callable
        The function to wrap.

    Returns
    -------
    callable
        Wrapped function where all arguments are guaranteed to be lists.

    Examples
    --------
    >>> @ensure_lists
    ... def foo(a, b):
    ...     return a, b
    >>> foo(1, [2, 3])
    ([1], [2, 3])
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any):
        # Modify the positional arguments to ensure each is a list
        new_args = [arg if isinstance(arg, list) else [arg] for arg in args]

        # Ensure the keyword arguments
        new_kwargs = {
            key: (value if isinstance(value, list) else [value])
            for key, value in kwargs.items()
        }

        return func(*new_args, **new_kwargs)

    return wrapper

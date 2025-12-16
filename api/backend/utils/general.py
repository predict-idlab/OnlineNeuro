# /api/backend/utils/general.py
from collections.abc import Iterable
from typing import IO

try:
    import psutil
except Exception:
    psutil = None


def debug_tree(pid: int, level: int = 0) -> None:
    """Print a safe, best-effort tree print. Uses psutil if available, else uses /proc (Linux).
    Attempts to use ``psutil`` for detailed child-process enumeration.
    If ``psutil`` is unavailable, falls back to a minimal Linux ``/proc``-based
    listing. All errors are caught and printed without raising.

    Parameters
    ----------
    pid : int
        The process ID from which to start printing the process tree.
    level : int, optional
        Initial indentation level used when printing the tree. Default is ``0``.

    Returns
    -------
    None

    """
    try:
        if psutil:
            p = psutil.Process(pid)

            def _print_proc(p, lvl=0):
                print("  " * lvl + f"- PID {p.pid}: {p.name()}")
                for c in p.children(recursive=False):
                    _print_proc(c, lvl + 1)

            _print_proc(p, level)
        else:
            # fallback: print immediate children via /proc (linux-only)
            print(f"- PID {pid} (no psutil available for detailed tree)")
    except Exception as e:
        print("debug_tree error:", e)


def close_flush_files(files: Iterable[IO]) -> None:
    """
    Close and flush a collection of file-like objects.

    Parameters
    ----------
    files : Iterable[IO]
        An iterable of open file-like objects. Each element must support
        ``flush()`` and ``close()`` methods.

    Returns
    -------
    None

    Notes
    -----
    Any exception raised during flushing or closing an individual file
    is silently ignored.

    """
    for f in files:
        try:
            f.flush()
            f.close()
        except Exception:
            pass


def collapse_lists(data: dict, max_depth: int | None = None) -> dict:
    """
    Recursively collapse lists of length one in a nested dictionary.

    Lists containing dictionaries with keys ``'min_value'`` or ``'max_value'``
    are preserved. The collapsing can be limited to a maximum recursion depth.

    Parameters
    ----------
    data : dict
        The nested dictionary to process. Only dictionaries are allowed; other
        types will raise a ``ValueError``.
    max_depth : int, optional
        Maximum depth to apply collapsing. If ``None``, collapse at all levels.
        Default is ``None``.

    Returns
    -------
    dict
        A new dictionary with lists of length one collapsed to their single
        element, preserving other lists and non-list values.
    """
    if not isinstance(data, dict):
        raise ValueError("Input must be a dictionary")
    result = {}
    for key, value in data.items():
        if isinstance(value, dict):
            if max_depth is None or max_depth > 0:
                result[key] = collapse_lists(
                    value, None if max_depth is None else max_depth - 1
                )
            else:
                result[key] = value

        elif isinstance(value, list):
            # Preserve list if it contains dicts with 'min_value' or 'max_value'
            if any(
                isinstance(item, dict) and ("min_value" in item or "max_value" in item)
                for item in value
            ):
                result[key] = value
            elif len(value) == 1:
                result[key] = value[0]
            else:
                result[key] = value
        else:
            result[key] = value

    return result

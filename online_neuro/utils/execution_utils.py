# online_neuro/utils/execution_utils.py
import json
import queue
import subprocess
import sys
from pathlib import Path

from .matlab_utils import run_matlab

__all__ = ["run_matlab"]


def run_python_script(
    script_path: str | Path,
    function_name: str = "main.py",
    verbose: bool = True,
    **kwargs,
) -> subprocess.Popen:
    """
    Launches an external Python script as a subprocess, passing configuration
    and connection details via command-line arguments.

    The script path is resolved relative to the root of the project structure
    (assumed to be two directories above the current file's location:
    two levels up from `networking/connection_utils.py`).

    Parameters
    ----------
    script_path : str | Path
        The path to the directory containing the main script (e.g., 'simulators/python').
        Can be absolute or relative to the project root.
    function_name : str, optional
        The name of the Python file to execute within `script_path`. Defaults to "main.py".
    verbose : bool, optional
        If True, prints additional information. Defaults to True.
    **kwargs : Any
        Keyword arguments containing configurations (like connection details, problem setup).
        These are converted into '--key JSON_STRING' or '--key VALUE' command line arguments.

    Returns
    -------
    subprocess.Popen
        The Popen object representing the launched external Python process.

    Raises
    ------
    FileNotFoundError
        If the resolved path to the script directory does not exist.
    Exception
        Catches and prints errors encountered during process launch (e.g., if 'python3' is not found).
    """
    try:
        parent_directory = Path(__file__).resolve().parents[2]
    except IndentationError:
        msg = "Parent directory not found at {__file__}"
        raise IndentationError(msg)

    sript_path = Path(script_path)

    # 1. Resolve the full path to the script directory
    if sript_path.is_absolute():
        script_full_path = script_path
    else:
        script_full_path = parent_directory / script_path

    script_full_path = Path(script_full_path)

    if not script_full_path.exists():
        raise FileNotFoundError(f"The Python script does not exist: {script_full_path}")

    if verbose:
        print(f"Calling Python script {script_path}")

    command = [sys.executable, str(script_full_path / function_name)]
    for key, value in kwargs.items():
        command.append(f"--{key}")
        if isinstance(value, dict):
            command.append(json.dumps(value))
        else:
            command.append(str(value))

    try:
        process = subprocess.Popen(command)
        return process

    except Exception as e:
        # TODO this needs to be handled or passed to the UI
        print("Error starting Python (process)")
        print(f"Details: {e}")
        raise


def monitor_output(output_queue: queue.Queue):
    """
    Continuously read output from a given queue and prints the content to the console.

    This function is designed to run in a separate thread (e.g., a Daemon thread)
    to asynchronously capture and display output streams (stdout/stderr) from a subprocess.

    Parameters
    ----------
    output_queue : queue.Queue
        The queue instance holding output lines (strings or bytes) piped from an
        external process.

    Returns
    -------
    None
        This function runs indefinitely until the thread is manually stopped or
        the main program exits (if run as a daemon thread).
    """
    while True:
        try:
            line = output_queue.get(timeout=1)  # Timeout to allow for graceful shutdown
            print(line, end="")  # Print each line from the queue
        except queue.Empty:
            continue  # No output available, keep checking

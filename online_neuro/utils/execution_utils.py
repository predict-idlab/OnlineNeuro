import json
import queue
import subprocess
from pathlib import Path

from .matlab_utils import run_matlab

__all__ = ["run_matlab"]


def run_python_script(
    script_path, function_name="main.py", verbose: bool = True, **kwargs
):
    """
    @param script_path:
    @param function_name:
    @param kwargs:
    @return:
    """

    current_directory = Path(__file__).resolve().parent
    parent_directory = current_directory.parent

    sript_path = Path(script_path)
    if sript_path.is_absolute():
        script_full_path = script_path
    else:
        script_full_path = parent_directory / script_path

    if not script_full_path.exists():
        raise FileNotFoundError(f"The Python script does not exist: {script_full_path}")

    if verbose:
        print(f"Calling Python script {script_path}")

    command = ["python3", str(script_full_path / function_name)]
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
        print(f"Error starting Python (process): {e}")


def monitor_output(output_queue):
    """Continuously reads from the output queue and prints lines as they come in."""
    while True:
        try:
            line = output_queue.get(timeout=1)  # Timeout to allow for graceful shutdown
            print(line, end="")  # Print each line from the queue
        except queue.Empty:
            continue  # No output available, keep checking

import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

from api.backend.utils.general import collapse_lists, debug_tree
from api.backend.utils.string_utils import convert_strings_to_numbers
from api.frontend.components.config_forms import (
    experiments_types,
    matlab_experiments,
    python_experiments,
)

try:
    import psutil
except Exception:
    psutil = None


def prepare_experiment(data: dict, port: int) -> list[str]:
    """
    Validate input data and construct/run the experiment subprocess.

    Parameters
    ----------
    data : dict
        Experiment, model, pulse, and optimizer parameters.
    port : int
        Flask server port to pass to the experiment.

    Returns
    -------
    list[str]
        The command executed (for logging/response purposes).
    """
    data = collapse_lists(data)

    if "experiment" not in data["experimentParameters"]:
        keys = list(data["experimentParameters"].keys())
        raise NotImplementedError(f"No experiment in {keys}")

    exp_orig = data["experimentParameters"]["experiment"]
    if exp_orig in python_experiments:
        exp_name = python_experiments[exp_orig]
        target = "Python"
    elif exp_orig in matlab_experiments:
        exp_name = matlab_experiments[exp_orig]
        target = "MATLAB"
    else:
        raise Exception(f"Experiment not defined: {exp_orig}")

    data["experimentParameters"]["problem_name"] = exp_name
    data["experimentParameters"]["problem_type"] = experiments_types[exp_name]

    main_path = str(Path("api") / "experiment_runner.py")
    base_command = ["python3", main_path]
    connection_payload = json.dumps({"target": target, "port_flask": str(port)})

    experiment_params = convert_strings_to_numbers(data.get("experimentParameters", {}))
    pulse_params = convert_strings_to_numbers(data.get("pulseParameters", {}))
    prob_load = json.dumps(
        {
            "experiment_parameters": experiment_params,
            "pulse_parameters": pulse_params,
        }
    )

    model_load = convert_strings_to_numbers(data.get("modelParameters", {}))
    model_load = json.dumps(model_load)

    optimizer_load = convert_strings_to_numbers(data.get("optimizerParameters", {}))
    optimizer_load = json.dumps(optimizer_load)

    command = (
        base_command
        + ["--problem_config", prob_load]
        + ["--connection_config", connection_payload]
        + ["--model_config", model_load]
        + ["--optimizer_config", optimizer_load]
    )

    return command


def stop_process(
    process, close_call: Callable | None = None, debug_tree_bool: bool = True
) -> str:
    """
    Attempt to gracefully stop a subprocess and its descendants.

    Parameters
    ----------
    process : subprocess.Popen
        The process to stop.
    close_call : Callable, optional
        Function to call when terminating if `process._log_files` exists.
    debug_tree_bool : bool, optional
        Whether to print the process tree for debugging.

    Returns
    -------
    str
        Status of the stop operation: 'stopped', 'forcefully stopped', or 'not running'.
    """
    if process is None or process.poll() is not None:
        if process is not None and hasattr(process, "_log_files") and close_call:
            close_call(process._log_files)
        return "not running"

    if debug_tree_bool:
        print("Process tree before kill:")
        debug_tree(process.pid)

    try:
        try:
            pgid = os.getpgid(process.pid)
        except Exception:
            pgid = None

        # polite termination
        try:
            if pgid is not None:
                os.killpg(pgid, signal.SIGTERM)
            else:
                process.terminate()
        except Exception as e:
            print("SIGTERM attempt failed:", e)

        # wait a short grace period
        waited = 0.0
        grace = 2.0
        interval = 0.1
        while waited < grace and process.poll() is None:
            time.sleep(interval)
            waited += interval

        if process.poll() is None:
            # force kill
            try:
                if pgid is not None:
                    print("Sending SIGKILL to PGID", pgid)
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    process.kill()
            except Exception as e:
                print("SIGKILL attempt failed:", e)

            waited2 = 0.0
            max_wait2 = 2.0
            while waited2 < max_wait2 and process.poll() is None:
                time.sleep(interval)
                waited2 += interval

        # psutil cleanup for descendants
        if process.poll() is None and psutil:
            try:
                parent = psutil.Process(process.pid)
                for child in parent.children(recursive=True):
                    try:
                        print("killing descendant", child.pid, child.name())
                        child.kill()
                    except Exception as e:
                        print("failed killing descendant:", e)
                try:
                    parent.kill()
                except Exception:
                    pass
            except Exception as e:
                print("psutil cleanup error:", e)

        # cleanup logs
        if hasattr(process, "_log_files") and close_call:
            close_call(process._log_files)

        # final status
        if process.poll() is not None:
            return "forcefully stopped" if process.returncode != 0 else "stopped"
        return "forcefully stopped"

    except Exception as e:
        print("Stop error:", e)
        try:
            if process and hasattr(process, "_log_files") and close_call:
                close_call(process._log_files)
        except Exception:
            pass
        return "unexpected error"


def monitor_process(
    proc: subprocess.Popen, process_ref: dict, lock: threading.Lock
) -> None:
    """
    Monitor a subprocess and clear the reference when it terminates.

    Parameters
    ----------
    proc : subprocess.Popen
        The subprocess to monitor.
    process_ref : dict
        A dictionary holding the 'process' reference, e.g. {'process': proc}.
        Using a dict allows it to be mutable and shared across threads/files.
    lock : threading.Lock
        Lock to synchronize access to the process_ref.
    """
    try:
        while True:
            if proc.poll() is not None:
                break
            time.sleep(0.25)
    except Exception as e:
        print("monitor_process exception:", e)
    finally:
        with lock:
            # Only clear if it's the same object (avoid races)
            if process_ref.get("process") is proc:
                process_ref["process"] = None
        print("monitor_process: process ended and cleaned up")

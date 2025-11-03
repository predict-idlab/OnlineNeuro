# api/app.py
import argparse
import io
import json
import os
import signal
import subprocess
import threading
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
from flask import Flask, Response, jsonify, render_template, request
from flask_socketio import SocketIO
from frontend.components.config_forms import (
    config_function,
    config_model,
    config_optimizer,
    config_problem,
    experiments_types,
    matlab_experiments,
    model_map,
    python_experiments,
)

# TODO improve typing to prevent Nones
start_time = time.time()
app = Flask(
    __name__,
    template_folder=str(Path("frontend") / "templates"),
    static_folder=str(Path("frontend") / "static"),
)

socketio = SocketIO(app, async_mode="eventlet")

DEFAULT_PORT = 10000

process = None
process_lock = threading.Lock()  # Lock to ensure thread safety

experiment_list = list(matlab_experiments.keys()) + list(python_experiments.keys())
model_list = list(model_map.keys())


def convert_strings_to_numbers(data) -> Any:
    if isinstance(data, dict):
        # Recursively apply the function to each value in the dictionary
        return {key: convert_strings_to_numbers(value) for key, value in data.items()}
    elif isinstance(data, list):
        # Recursively apply the function to each element in the list
        return [convert_strings_to_numbers(item) for item in data]
    elif isinstance(data, str):
        # Normalize the string to lowercase for boolean checks
        lower_data = data.lower()

        # Handle boolean strings 'true' and 'false'
        if lower_data == "true":
            return True
        elif lower_data == "false":
            return False

        # Try to convert the string to an int or float
        try:
            if "." in data:  # Convert to float if the string contains a decimal point
                return float(data)
            else:
                return int(data)  # Convert to int otherwise
        except ValueError:
            return data  # Return the original string if conversion fails
    else:
        # Return the value if it is not a string, dict, or list
        return data


def monitor_process(proc):
    proc.wait()
    with process_lock:
        global process
        process = None


@app.route("/")
def index():
    port = app.config.get("PORT", DEFAULT_PORT)  # Default to 9000 if not set
    return render_template("frontend.html", port=port)


@app.route("/experiments", methods=["GET"])
def get_experiments():
    return jsonify(experiment_list)


@app.route("/models", methods=["GET"])
def get_models():
    return jsonify(model_list)


@app.route("/get_fun_parameters", methods=["POST"])
def get_fun_parameters():
    data = request.json
    if data is None:
        return jsonify({"error": "No data provided"}), 400

    function = data.get("function")

    if not function:
        return jsonify({"error": "No experiment provided"}), 400
    try:
        fun_config = config_function(function)
        return jsonify(fun_config)
    except Exception as e:
        msg = str(e)
        return jsonify({"error": msg}), 400


@app.route("/get_parameters", methods=["POST"])
def get_parameters():
    data = request.json

    if data is None:
        return jsonify({"error": "No data provided"}), 400

    parameters_type = data.get("type")
    print(data)
    if not parameters_type:
        return jsonify({"error": "No parameters' type provided"}), 400

    if parameters_type == "problem_parameters":
        experiment = data.get("experiment")
        if not experiment:
            return jsonify({"error": "No experiment provided"}), 400

        if experiment in matlab_experiments:
            params = config_problem(matlab_experiments[experiment])
        elif experiment in python_experiments:
            params = config_problem(python_experiments[experiment])
        else:
            raise NotImplementedError
    elif parameters_type == "model":
        model = data.get("model")
        params = config_model(model)

    elif parameters_type == "optimizer":
        params = config_optimizer()

    else:
        raise NotImplementedError

    return jsonify(params)


@app.route("/plot", methods=["GET"])
def plot():
    # Get the port number from query parameters
    default_port = app.config.get("PORT", DEFAULT_PORT)  # Default to 9000 if not set
    port = request.args.get("port", default=default_port, type=int)

    # Get the plot type from query parameters, defaulting to 'plotly'
    plot_type = request.args.get("type", default="plotly", type=str)

    # Choose the template based on the plot type
    if plot_type == "contour":
        return render_template("plot_contour.html", port=port)
    elif plot_type == "pulses":
        return render_template("plot_pulses.html", port=port)
    elif plot_type == "plotly":
        return render_template("plotly.html", port=port)
    else:
        warnings.warn(f"Not a valid plot {plot_type}")
        return f"Error: Invalid plot type '{plot_type}' requested.", 400


def collapse_lists(data, max_depth=None):
    """Collapse lists of length one in a nested dictionary.
    Args:
        data (dict): Input dictionary to process.
        max_depth (int, optional): Maximum depth to apply collapsing. If None, collapse at all levels.
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


@app.route("/start", methods=["POST"])
def prepare_experiment() -> Response | tuple[Response, int]:
    global process
    # global process_lock # TODO Probably not needed here?

    data = request.json

    if data is None:
        return jsonify({"error": "No data provided"}), 400

    if not isinstance(data, dict):
        try:
            data = json.loads(data)
        except Exception as e:
            return jsonify({"error": f"Invalid JSON format: {e}"}), 400

    print("Received data for experiment:")
    print(data)

    try:
        with process_lock:
            if process is not None:
                return jsonify({"status": "already running"})

            if "test_command" in data:
                command = data["test_command"]
                print(f"Running in TEST mode with command: {command}")
            else:

                data = collapse_lists(data)

                if "experiment" in data["experimentParameters"]:
                    exp_orig = data["experimentParameters"]["experiment"]
                    if exp_orig in python_experiments:
                        exp_name = python_experiments[exp_orig]
                    elif exp_orig in matlab_experiments:
                        exp_name = matlab_experiments[exp_orig]
                    else:
                        raise Exception(
                            f"Experiment with such name is not defined {data['experiment']}"
                        )
                    data["experimentParameters"]["problem_name"] = exp_name
                    data["experimentParameters"]["problem_type"] = experiments_types[
                        exp_name
                    ]
                else:
                    keys = list(data["experimentParameters"].keys())
                    raise NotImplementedError(f"No experiment in {keys}")

                print("reformatted experiment:")
                print(data)

                port = app.config.get(
                    "PORT", DEFAULT_PORT
                )  # Default to 9000 if not set
                main_path = str(Path("api") / "experiment_runner.py")
                base_command = ["python3", main_path]

                if exp_orig in matlab_experiments:
                    connection_payload = {"target": "MATLAB", "port_flask": str(port)}
                elif exp_orig in python_experiments:
                    connection_payload = {"target": "Python", "port_flask": str(port)}
                else:
                    raise NotImplementedError

                connection_payload = json.dumps(connection_payload)
                experiment_params = convert_strings_to_numbers(
                    data.get("experimentParameters", {})
                )
                pulse_params = convert_strings_to_numbers(
                    data.get("pulseParameters", {})
                )

                problem_config_dict = {
                    "experiment_parameters": experiment_params,
                    "pulse_parameters": pulse_params,
                }
                prob_load = json.dumps(problem_config_dict)

                model_load = convert_strings_to_numbers(data.get("modelParameters", {}))
                model_load = json.dumps(model_load)

                command = (
                    base_command
                    + ["--problem_config", prob_load]
                    + ["--connection_config", connection_payload]
                    + ["--model_config", model_load]
                )

            preexec_fn = os.setsid if hasattr(os, "setsid") else None
            process = subprocess.Popen(command, preexec_fn=preexec_fn)
            # Start the process with the given command
            threading.Thread(
                target=monitor_process, args=(process,), daemon=True
            ).start()  # Monitor the process

            return jsonify({"status": "started", "command": command})

    except Exception as e:
        # Handle exceptions
        print(f"An unexpected error occurred: {e}")  # Optionallfy log the error
        traceback.print_exec()
        return jsonify({"status": "Unexpected error", "error": str(e)}), 500


@app.route("/status", methods=["GET"])
def check_experiment() -> Response | tuple[Response, int]:
    global process
    try:
        with process_lock:
            if process is None or process.poll() is not None:
                return jsonify({"status": "not running"})
            return jsonify({"status": "running"})
    except Exception as e:
        print(f"An unexpected error occurred: {e}")  # Optionally log the error
        return jsonify({"status": "Unexpected error", "error": str(e)}), 500


# Function to handle stopping the experiment
@app.route("/stop", methods=["POST"])
def stop_experiment() -> Response | tuple[Response, int]:
    global process
    try:
        with process_lock:
            if process is not None and process.poll() is None:
                print(f"Attempting to stop process with PID: {process.pid}")
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)

                except ProcessLookupError:
                    # Process ended
                    pass

                except AttributeError:
                    process.terminate()
                    print("Sent SIGTERM to process (Windows fallback)")

                try:
                    # Waiting for n seconds for shutdown
                    process.wait(timeout=3)  # Ensure the process has finished
                    return jsonify({"status": "stopped"})

                except subprocess.TimeoutExpired:
                    print("process did not terminate. Forcking kill.")
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                        print("sent SIGKILL to processs")
                    except ProcessLookupError:
                        # Process may have ended
                        pass
                    except AttributeError:
                        process.kill()
                        print("Sent SIGKILL to process (Windows fallback)")

                    process.wait()
                    process = None
                    return jsonify({"status": "Forcefully stopped"})

            # Process already ended (shouldn't occur)
            if process is not None and process.poll() is not None:
                process = None  # Clearing up

            return jsonify({"status": "not running"})

    except Exception as e:
        print(f"An unexpected error occurred: {e}")  # Optionally log the error
        traceback.print_exc()
        return jsonify({"status": "Unexpected error", "error": str(e)}), 500


@app.route("/update_data", methods=["POST"])
def update_data():
    """Receive new data and transmit it to the plot
    TODO extend this to handle other plot types
    """
    data = request.get_json()
    plot_data = data.get("data")
    plot_type = data.get("plot_type")

    try:
        if isinstance(plot_data, str):
            # TODO replace this to prevent pandas deprecation warning
            plot_data = pd.read_json(io.StringIO(plot_data))

        if plot_type == "pairplot":
            first_column = plot_data.columns[0]
            second_column = plot_data.columns[1]
            socketio.emit(
                "pairplot",
                {
                    "x": plot_data[first_column].tolist(),
                    "y": plot_data[second_column].tolist(),
                    "class": plot_data["response"].tolist(),
                },
            )

        elif plot_type == "contour":
            first_column = plot_data.columns[0]
            second_column = plot_data.columns[1]
            z = data.get("z")
            socketio.emit(
                "contour",
                {
                    "x": plot_data[first_column].tolist(),
                    "y": plot_data[second_column].tolist(),
                    "z": z,
                },
            )

        elif plot_type == "pulses":
            socketio.emit(
                "pulses",
                {
                    "t": plot_data.get("time").tolist(),
                    "y0": plot_data.get("pulse_a").tolist(),
                    "y1": plot_data.get("pulse_b").tolist(),
                },
            )
        else:
            return jsonify({"message": f"Plot type {plot_type} not implemented"}), 501
        # Return a success response
        return jsonify({"message": "Data updated successfully"}), 200

    except Exception as e:
        # Handle exceptions and return an error response
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    # Start the background thread to check the cache for new data
    print("Starting Flask app...")
    parser = argparse.ArgumentParser(description="Process arguments")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Flask port.")
    args = parser.parse_args()

    print(f"http://localhost:{args.port}")
    socketio.run(app, port=args.port, debug=False)

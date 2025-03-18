# frontend/app.py
import json
from common.utils import load_json
from flask import Flask, request, jsonify, render_template
import threading
import time
import subprocess
import argparse
import traceback
import pandas as pd
from frontend.components.config_forms import (
    config_problem, config_function, python_experiments,
    experiments_types, matlab_experiments,
    config_model, model_map
)
from flask_socketio import SocketIO
from pathlib import Path
import warnings

# Added to prevent unused warning
start_time = time.time()
app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet')

FLASK_PORT = 9000

process = None
data_cache = None
cache = None

process_lock = threading.Lock()  # Lock to ensure thread safety
cache_lock = threading.Lock()

experiment_list = list(matlab_experiments.keys()) + list(python_experiments.keys())
model_list = list(model_map.keys())


def convert_strings_to_numbers(data):
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
        if lower_data == 'true':
            return True
        elif lower_data == 'false':
            return False

        # Try to convert the string to an int or float
        try:
            if '.' in data:  # Convert to float if the string contains a decimal point
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


@app.route('/')
def index():
    return render_template('frontend.html', port=FLASK_PORT)


@app.route('/experiments', methods=['GET'])
def get_experiments():
    return jsonify(experiment_list)


@app.route('/models', methods=['GET'])
def get_models():
    return jsonify(model_list)


@app.route('/get_fun_parameters', methods=['POST'])
def get_fun_parameters():
    data = request.json
    function = data.get('function')
    if not function:
        return jsonify({'error': 'No experiment provided'}), 400
    try:
        fun_config = config_function(function)
        return jsonify(fun_config)
    except Exception as e:
        msg = str(e)
        return jsonify({'error': msg}), 400


@app.route('/get_parameters', methods=['POST'])
def get_parameters():
    data = request.json
    parameters_type = data.get('type')

    if not parameters_type:
        return jsonify({'error': "No parameters' type provided"}), 400

    if parameters_type == 'problem_parameters':
        experiment = data.get('experiment')
        if not experiment:
            return jsonify({'error': 'No experiment provided'}), 400

        if experiment in matlab_experiments:
            params = config_problem(matlab_experiments[experiment])
        elif experiment in python_experiments:
            params = config_problem(python_experiments[experiment])
        else:
            raise NotImplementedError

    elif parameters_type == 'model_parameters':
        model = data.get('model')
        params = config_model(model)
    else:
        raise NotImplementedError

    return jsonify(params)


@app.route('/plot', methods=['GET'])
def plot():
    # Get the port number from query parameters
    port = request.args.get('port', default=FLASK_PORT, type=int)

    # Get the plot type from query parameters, defaulting to 'plotly'
    plot_type = request.args.get('type', default='plotly', type=str)

    # Choose the template based on the plot type
    if plot_type == 'contour':
        return render_template('plot_contour.html', port=port)
    elif plot_type == 'pulses':
        return render_template('plot_pulses.html', port=port)
    elif plot_type == 'plotly':
        return render_template('plotly.html', port=port)
    else:
        warnings.warn(f'Not a valid plot {plot_type}')
        pass


def collapse_lists(data, max_depth=0):
    if not isinstance(data, dict):
        raise ValueError('Input must be a dictionary')

    result = {}
    for key, value in data.items():
        if isinstance(value, dict):
            if max_depth > 0:
                result[key] = collapse_lists(value, max_depth - 1)
            else:
                result[key] = value
        elif isinstance(value, list):
            # Check if any dictionary inside the list contains 'min_value' or 'max_value'
            if any(isinstance(item, dict) and ('min_value' in item or 'max_value' in item) for item in value):
                result[key] = value  # Preserve the list
            elif len(value) == 1:
                result[key] = value[0]  # Collapse single-element list
            else:
                result[key] = value  # Keep multi-element lists unchanged
        else:
            result[key] = value

    return result


@app.route('/start', methods=['POST'])
def prepare_experiment():
    global process
    global process_lock

    data = request.json
    # TODO Quick fix. make this more robust later

    if 'pulse_parameters' not in data:
        data = data['other_parameters']
    else:
        for k, v in data['other_parameters'].items():
            if k not in data.keys():
                if isinstance(v, dict):
                    data[k] = [v]
                else:
                    data[k] = v
            else:
                raise Exception(f"Key {k} is duplicated in data['other_parameters']")
        del data['other_parameters']

    data = collapse_lists(data)

    if 'experiment' in data:
        exp_orig = data['experiment']
        if data['experiment'] in python_experiments:
            exp_name = python_experiments[exp_orig]
        elif data['experiment'] in matlab_experiments:
            exp_name = matlab_experiments[exp_orig]
        else:
            raise Exception(f"Experiment with such name is not defined {data['experiment']}")

        data['experiment'] = dict()
        data['experiment']['name'] = exp_name
        data['experiment']['type'] = experiments_types[exp_name]
    else:
        keys = list(data.keys())
        raise NotImplementedError(f'No experiment in {keys}')

    # TODO this call will be eventually moved to the frontend and not online_neuro
    main_path = str(Path('online_neuro') / 'server_side.py')
    base_command = ['python3', main_path]

    if exp_orig in matlab_experiments:
        connection_payload = {
            'target': 'MATLAB',
            'port_flask': str(FLASK_PORT)
        }
    elif exp_orig in python_experiments:
        connection_payload = {
            'target': 'Python',
            'port_flask': str(FLASK_PORT)
        }
    else:
        raise NotImplementedError

    connection_payload = json.dumps(connection_payload)
    prob_load = convert_strings_to_numbers(data)
    prob_load = json.dumps(prob_load)

    command = base_command + ['--problem_config', prob_load] + ['--connection_config', connection_payload]

    try:
        with process_lock:
            if process is None:
                print('Command')
                print(command)
                process = subprocess.Popen(command)  # Start the process with the given command
                threading.Thread(target=monitor_process, args=(process,),
                                 daemon=True).start()  # Monitor the process
                return jsonify({'status': 'started', 'command': command})
            else:
                return jsonify({'status': 'already running'})

    except Exception as e:
        # Handle exceptions
        print(f'An unexpected error occurred: {e}')  # Optionally log the error
        traceback.print_exec()
        return jsonify({'status': 'Unexpected error', 'error': str(e)}), 500


@app.route('/status', methods=['GET'])
def check_experiment():
    global process
    try:
        with process_lock:
            if process is None or process.poll() is not None:
                return jsonify({'status': 'not running'})
            return jsonify({'status': 'running'})
    except Exception as e:
        print(f'An unexpected error occurred: {e}')  # Optionally log the error
        return jsonify({'status': 'Unexpected error', 'error': str(e)}), 500


# Function to handle stopping the experiment
@app.route('/stop', methods=['POST'])
def stop_experiment() -> None:
    global process
    try:
        with process_lock:
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                    process.wait()  # Ensure the process has finished
                    process = None
                    return jsonify({'status': 'stopped'})
                else:
                    process = None
                    return jsonify({'status': 'Process had already ended'})

            return jsonify({'status': 'not running'})
    except Exception as e:
        print(f'An unexpected error occurred: {e}')  # Optionally log the error
        return jsonify({'status': 'Unexpected error', 'error': str(e)}), 500


@app.route('/get_data')
def get_data():
    global cache
    """Return the full dataset, so it can be used for plotting."""
    return jsonify(cache)


@app.route('/update_data', methods=['POST'])
def update_data():
    """Receive new data and transmit it to the plot
    TODO extend this to handle the different plots?
    """
    data = request.get_json()
    plot_data = data.get('data')
    plot_type = data.get('plot_type')
    try:
        if isinstance(plot_data, str):
            # TODO replace this to prevent pandas deprecation warning
            plot_data = pd.read_json(plot_data)

        if plot_type == 'pairplot':
            first_column = plot_data.columns[0]
            second_column = plot_data.columns[1]

            socketio.emit('pairplot', {
                'x': plot_data[first_column].tolist(),
                'y': plot_data[second_column].tolist(),
                'class': plot_data['response'].tolist()
            })

        elif plot_type == 'contour':
            first_column = plot_data.columns[0]
            second_column = plot_data.columns[1]
            z = data.get('z')
            socketio.emit('contour', {
                'x': plot_data[first_column].tolist(),
                'y': plot_data[second_column].tolist(),
                'z': z,
            })

        elif plot_type == 'pulses':
            socketio.emit('pulses', {
                't': plot_data.get('time').tolist(),
                'y0': plot_data.get('pulse_a').tolist(),
                'y1': plot_data.get('pulse_b').tolist(),
            })

        else:
            return jsonify({'message': f'Plot type {plot_type} not implemented'}), 501
        # Return a success response
        return jsonify({'message': 'Data updated successfully'}), 200

    except Exception as e:
        # Handle exceptions and return an error response
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    # Start the background thread to check the cache for new data
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('--port', type=int, default=9000, help='Flask port.')
    args = parser.parse_args()

    print(f'http://localhost:{args.port}')
    socketio.run(app, port=args.port, debug=False)

# frontend/app.py
import json
import numpy as np
from flask import Flask, request, jsonify, render_template
import threading
import time
import subprocess
import atexit
import requests
import argparse
import traceback

from frontend.components.config_forms import config_problem

app = Flask(__name__)
#CORS(app)  # This will allow CORS for all routes and origins

DASK_PORT = 9000
DASK_URL = f"http://localhost:{DASK_PORT}"

process = None
data_cache = None

cache = {'x': [], 'y': []}
x_ix = 0

process_lock = threading.Lock()  # Lock to ensure thread safety
cache_lock = threading.Lock()

matlab_experiments = {"Axonsim (nerve block)": "axonsim_nerve_block",
                       "Axonsim (regression)": "axonsim_regression",
                       "Toy Regression": "rose_regression",
                       "Toy Classification": "circle_classification",
                       "Toy MOO": "vlmop2",
                       "Placeholder": "placeholder"}

python_experiments = {}

experiment_list = list(matlab_experiments.keys()) + list(python_experiments.keys())

def monitor_process(proc):
    proc.wait()
    with process_lock:
        global process
        process = None

def generate_data():
    # Simulate adding new data points (this could come from real sources)
    global cache
    while True:
        x_val = 0 if len(cache['x'])==0 else cache['x'][-1]+1
        cache['x'].append(x_val)
        cache['y'].append(np.random.uniform(0, 10))
        print(cache)
        print("data extended")
        #eventlet.sleep(1)



@app.route('/')
def index():
    return render_template('frontend.html', port=DASK_PORT)


@app.route('/experiments', methods=['GET'])
def get_experiments():
    return jsonify(experiment_list)

@app.route('/get_parameters', methods=['POST'])
def get_parameters():
    data = request.json
    experiment = data.get('experiment')
    if not experiment:
        return jsonify({"error": "No experiment provided"}), 400

    exp_params = config_problem(experiment)
    return jsonify(exp_params)


# def start_flask() -> None:
#     """
#     Start Flask backend to handle calls to optimizers, plotting, etc...
#     @return:  None
#     """
#     global flask_process
#     global PORT
#     try:
#         if flask_process is None or flask_process.poll() is not None:  # Check if process has finished
#             print("Starting process manager")
#             flask_process = subprocess.Popen(['python3', 'backend/process_manager.py', '--port', str(PORT)],
#                                              stdout=subprocess.PIPE, stderr=subprocess.PIPE
#                                              )
#             time.sleep(1)
#         else:
#             print("Flask is already running.")
#     except Exception as e:
#         raise e
#
#
# def kill_flask() -> None:
#     """
#     Make sure Flask server is terminated before finishing the process.
#     @return: None
#     """
#     global flask_process
#     if flask_process is not None:
#         flask_process.terminate()  # Attempt to terminate the process
#         try:
#             flask_process.wait(timeout=5)  # Wait for the process to terminate
#         except subprocess.TimeoutExpired:
#             print("Flask process did not terminate in time, forcing kill.")
#             flask_process.kill()  # Force kill
#         finally:
#             flask_process = None  # Set to None
#     else:
#         print("No process manager running.")


# Fetch the plot from Flask
@app.route('/plot', methods=['GET'])  # Use GET here
def plot():
    port = request.args.get('port', default=DASK_PORT, type=int)  # Get dark mode from query parameter
    return render_template('plotly.html', port=port)


# Thread function to run the experiment
@app.route('/start', methods=['POST'])
def prepare_experiment():
    global process
    global process_lock

    data = request.json

    print(data['parameters'])
    if 'experiment' in data['parameters']:
        if data['parameters']['experiment'] in matlab_experiments:
            data['parameters']['experiment'] = matlab_experiments[data['parameters']['experiment']]
        elif data['parameters']['experiment'] in python_experiments:
            data['parameters']['experiment'] = python_experiments[data['parameters']['experiment']]
        else:
            raise Exception("Not simulator implemented (?)")

    if data['problem'] in matlab_experiments:
        base_command = ["python3", "online_neuro/server_side.py"]
        connection_payload = {
            'initiator': 'Python',
            'target': 'MATLAB'
        }

    else:
        raise NotImplementedError

    connection_payload = json.dumps(connection_payload)
    prob_load = json.dumps(data['parameters'])
    command = base_command + ["--problem_config", prob_load] + ["--connection_config", connection_payload]

    try:
        with process_lock:
            if process is None:
                process = subprocess.Popen(command)  # Start the process with the given command
                threading.Thread(target=monitor_process, args=(process,),
                                 daemon=True).start()  # Monitor the process
                return jsonify({'status': 'started', 'command': command})
            else:
                return jsonify({'status': 'already running'})

    except Exception as e:
        # Handle exceptions
        print(f"An unexpected error occurred: {e}")  # Optionally log the error
        traceback.print_exec()
        return jsonify({"status": "Unexpected error", "error": str(e)}), 500


@app.route('/status', methods=['GET'])
def check_experiment():
    global process
    try:
        with process_lock:
            if process is None or process.poll() is not None:
                return jsonify({'status': 'not running'})
            return jsonify({'status': 'running'})
    except Exception as e:
        print(f"An unexpected error occurred: {e}")  # Optionally log the error
        return jsonify({"status": "Unexpected error", "error": str(e)}), 500

# Function to handle stopping the experiment
@app.route('/stop', methods=['POST'])
def stop_experiment() -> None:
    global process
    try:
        with process_lock:
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                    process.join()  # Ensure the process has finished
                    process = None
                    return jsonify({'status': 'stopped'})
                else:
                    process = None
                    return jsonify({'status': 'Process had already ended'})

            return jsonify({'status': 'not running'})
    except Exception as e:
        print(f"An unexpected error occurred: {e}")  # Optionally log the error
        return jsonify({"status": "Unexpected error", "error": str(e)}), 500

### TODO the plotting

@app.route('/get_data')
def get_data():
    global cache
    """Return the full dataset, so it can be used for plotting."""
    return jsonify(cache)


@app.route('/update_data')
def update_data():
    """Generate new data and return the updated dataset."""
    global x_ix
    global cache
    new_data = dict()
    new_data['x'] = cache['x'][x_ix:]
    new_data['y'] = cache['y'][x_ix:]
    x_ix = len(cache['x'])

    print("pack", new_data)
    return jsonify(new_data)


@app.route('/add_to_cache', methods=['POST'])
def add_to_cache():
    """
    Endpoint to add new data to the cache.
    """
    global data_cache
    new_data = request.json.get('data',{})
    if new_data:
        with cache_lock:
            data_cache = new_data
        return jsonify({'status': 'New data added to cache'}), 200
    return jsonify({'error': 'No data provided'}), 400

if __name__ == '__main__':
    # Start the background thread to check the cache for new data
    parser = argparse.ArgumentParser(description="Process arguments")
    parser.add_argument('--port', type=int, help='Flask port.')
    args = parser.parse_args()

    app.run(port=DASK_PORT, debug=False)


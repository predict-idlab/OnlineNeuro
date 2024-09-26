# backend/process_manager.py
import os
import time
from flask import Flask, request, jsonify, send_file
import subprocess
import threading
import argparse
import matplotlib.pyplot as plt

#TODO replace matplotlib by a server using Plotly resampler?
app = Flask(__name__)
process = None
process_lock = threading.Lock()  # Lock to ensure thread safety
cache_lock = threading.Lock()
plot_path_0 = '.cache/generated_plot_0.png'
plot_path_1 = '.cache/generated_plot_1.png'
plot_path_2 = '.cache/generated_plot_2.png'
plot_path_3 = '.cache/generated_plot_3.png'


def monitor_process(proc):
    proc.wait()
    with process_lock:
        global process
        process = None


@app.route('/start', methods=['POST'])
def start_process():
    global process
    with process_lock:
        if process is None:
            data = request.get_json()  # Get the command from the POST body
            command = data.get('command')
            if command:
                process = subprocess.Popen(command.split())  # Start the process with the given command
                threading.Thread(target=monitor_process, args=(process,), daemon=True).start()  # Monitor the process
                return jsonify({'status': 'started', 'command': command})
            return jsonify({'error': 'No command provided'}), 400
        return jsonify({'status': 'already running'})


@app.route('/stop', methods=['POST'])
def stop_process():
    global process
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


@app.route('/status', methods=['GET'])
def check_status():
    global process
    with process_lock:
        if process is None or process.poll() is not None:
            return jsonify({'status': 'not running'})
        return jsonify({'status': 'running'})



# ### Plotting and Cache Functionality ###
def check_for_new_data():
    """
    Background thread to monitor the cache and generate a plot when new data is detected.
    """
    print("Checking for new data?")
    while True:
        with cache_lock:
            if data_cache is not None:  # Replace with your cache-check logic
                create_plot(data_cache)  # Call the plot function
                data_cache = None  # Clear the cache after creating the plot
        time.sleep(5)  # Polling interval


def create_plot(data):
    """
    Creates a plot based on the provided data and saves it to disk.
    """
    #TODO


@app.route('/plot', methods=['POST'])
def get_plot():
    """
    Flask route to serve the generated plot.
    """
    data = request.get_json()  # Get the command from the POST body
    plot_path = data.get('plot_path')

    if os.path.exists(plot_path):
        return send_file(plot_path, mimetype='image/png')
    return jsonify({'error': 'No plot available'}), 404


@app.route('/add_to_cache', methods=['POST'])
def add_to_cache():
    """
    Endpoint to add new data to the cache.
    """
    global data_cache
    new_data = request.json.get('data')
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

    threading.Thread(target=check_for_new_data, daemon=True).start()

    app.run(port=args.port)

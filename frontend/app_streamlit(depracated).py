# frontend/app.py
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import time
import subprocess
import atexit
import requests
from components.config_forms import config_problem

PORT = 8888
BASE_URL = f"http://localhost:{PORT}"
flask_process = None

st.session_state['flask_running'] = False


def start_flask() -> None:
    """
    Start Flask backend to handle calls to optimizers, plotting, etc...
    @return:  None
    """
    global flask_process
    global PORT
    if flask_process is None or flask_process.poll() is not None:  # Check if process has finished
        flask_process = subprocess.Popen(['python3', 'backend/process_manager.py', '--port', str(PORT)])
        st.session_state['flask_running'] = True
        time.sleep(3)
    else:
        print("Flask is already running.")


def kill_flask() -> None:
    """
    Make sure Flask server is terminated before finishing the process.
    @return: None
    """
    global flask_process
    if flask_process is not None:
        flask_process.terminate()  # Attempt to terminate the process
        try:
            flask_process.wait(timeout=5)  # Wait for the process to terminate
        except subprocess.TimeoutExpired:
            print("Flask process did not terminate in time, forcing kill.")
            flask_process.kill()  # Force kill
        finally:
            flask_process = None  # Set to None


atexit.register(kill_flask)  # Ensure the process is killed when the app exits

# Initialize session state for experiment control
if 'experiment_running' not in st.session_state:
    st.session_state['experiment_running'] = False

# Sidebar - Generate UI elements based on JSON keys
st.sidebar.title("Configuration")

experiment_list_map = {"Axonsim (nerve block)": "axonsim_nerve_block",
                       "Axonsim (regression)": "axonsim_regression",
                       "Toy Regression": "rose_regression",
                       "Toy Classification": "circle_classification",
                       "Toy MOO": "vlmop2",
                       "Placeholder": "placeholder"}

experiment_list = list(experiment_list_map.keys())
if "disable_selectbox" not in st.session_state:
    st.session_state['disable_selectbox'] = False  # Initially disable the selectbox
def enable_selectbox():
    st.session_state['disable_selectbox'] = False
def disable_selectbox():
    st.session_state['disable_selectbox'] = True

st.session_state['experiment'] = st.sidebar.selectbox("Experiment/simulation",
                                                      experiment_list,
                                                      index=None,
                                                      placeholder="Select experiment...",
                                                      disabled=st.session_state['disable_selectbox']
                                                      )
st.session_state['config'] = dict()
st.session_state['config']['problem'] = dict()

if st.session_state['experiment'] is not None:
    st.session_state['config']['problem'] = dict()
    st.session_state['config']['problem'] = config_problem(experiment_list_map[st.session_state['experiment']])


# Display updated configuration in the sidebar
st.sidebar.write("Current Configuration:",
                 st.session_state['config']['problem'])

# Main page
st.title("Live Data Plots")

# Create two columns for side-by-side plots
col1, col2 = st.columns(2)

# Placeholders for live plot updates
placeholder1 = col1.empty()
placeholder2 = col2.empty()

# Fetch the plot from Flask
def fetch_plot(dark_mode):
    # Adjust the URL based on your Flask server setup
    try:
        response = requests.get(f"{BASE_URL}/plot?dark_mode={dark_mode}?port={PORT}")
        return response.text
    except requests.ConnectionError:
        st.error("Failed to connect to the Flask server.")
        return None

# Thread function to run the experiment
def start_experiment_thread() -> None:
    check_experiment_thread()

    if not st.session_state['experiment_running']:
        if not st.session_state['flask_running']:
            start_flask()
            time.sleep(3)

        st.session_state['experiment_running'] = True
        disable_selectbox()
        state_copy = st.session_state['config'].copy()
        #TODO adjust this so it executes other targets

        command = ["python3 online_neuro/server_side.py", '--conn_initiator Python', '--conn_target MATLAB']
        for k, v in state_copy['problem'].items():
            command.append(f"--prob_{k} {v}")

        command = ' '.join(command)
        state_copy['command'] = command

        response = requests.post(f"{BASE_URL}/start", json=state_copy)
        response = response.json()
        print(response)
    else:
        print("Experiment already running")


def check_experiment_thread() -> None:
    if st.session_state['experiment_running']:
        response = requests.get(f"{BASE_URL}/status")
        response = response.json()
        print(response)

        # If process ended, set experiment_running to false
        if response['status'] != 'running':
            st.session_state['experiment_running'] = False

    else:
        enable_selectbox()
        print("No experiment is running")


# Function to handle stopping the experiment
def stop_experiment_thread() -> None:
    if st.session_state['experiment_running']:
        response = requests.post(f"{BASE_URL}/stop")
        response = response.json()
        print(response)
    else:
        enable_selectbox()
        print("No experiment is running")

    st.session_state['experiment_running'] = False


# Sidebar buttons to control experiment start/stop
st.sidebar.button("Start Experiment", on_click=start_experiment_thread)
st.sidebar.button("Check Experiment", on_click=check_experiment_thread)
st.sidebar.button("Stop Experiment", on_click=stop_experiment_thread)

if not st.session_state['flask_running']:
    start_flask()

with placeholder1:
    plot_url = f"{BASE_URL}/plot"
    st.components.v1.iframe(src=plot_url,
                            height=400,
                            scrolling=False)  # Adjust the height as needed



# frontend/app.py
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import time
import subprocess
import atexit
import requests

from components.config_forms import config_problem
PORT = 8888
BASE_URL = f"http://localhost:{PORT}"
flask_process = None
def start_flask()->None:
    global flask_process
    global PORT
    if flask_process is None or flask_process.poll() is not None:  # Check if process has finished
        flask_process = subprocess.Popen(['python3', 'backend/process_manager.py', '--port', str(PORT)])
    else:
        print("Flask is already running.")
def kill_flask()->None:
    global flask_process
    if flask_process is not None:
        flask_process.terminate()  # Attempt to terminate the process
        try:
            flask_process.wait(timeout=5)  # Wait for the process to terminate
        except subprocess.TimeoutExpired:
            print("Flask process did not terminate in time, forcing kill.")
            flask_process.kill()  # Force kill if it doesn't exit
        finally:
            flask_process = None  # Set to None after killing


atexit.register(kill_flask)  # Ensure the process is killed when the app exits


# Function to generate random data (simulating live data updates)
def generate_data( )->np.ndarray:
    return np.random.randn(100)

def experiment()->None:
    #Placeholder code
    st.session_state['data1'] = generate_data()
    st.session_state['data2'] = generate_data()

    # Sleep for a short while to simulate new data coming in
    time.sleep(2)
    update_plots()




# Initialize session state for experiment control
if 'experiment_running' not in st.session_state:
    st.session_state['experiment_running'] = False

if 'data1' not in st.session_state:
    st.session_state['data1'] = np.zeros(100)  # Initialize with zeros
if 'data2' not in st.session_state:
    st.session_state['data2'] = np.zeros(100)  # Initialize with zeros

# Sidebar - Generate UI elements based on JSON keys
st.sidebar.title("Configuration")

st.session_state['experiment'] = st.sidebar.selectbox("Experiment/simulation",
                                                      ("Axonsim (nerve block)", "Axonsim (regression)", "Toy Regression",
                                                       "Toy Classification", "Toy MOO", "Placeholder"),
                                                      index=None,
                                                      placeholder="Select experiment...",
)

if st.session_state['experiment'] is not None:
    config = config_problem(st.session_state['experiment'])
else:
    config = dict()

# Display updated configuration in the sidebar
st.sidebar.write("Current Configuration:", config)

# Main page - Display two plots side by side
st.title("Live Data Plots")

# Create two columns for side-by-side plots
col1, col2 = st.columns(2)

# Placeholders for live plot updates
placeholder1 = col1.empty()
placeholder2 = col2.empty()


# Thread function to run the experiment
def start_experiment_thread() -> None:
    check_experiment_thread()

    if not st.session_state['experiment_running']:
        st.session_state['experiment_running'] = True
        start_flask()
        time.sleep(1)
        response = requests.post(f"{BASE_URL}/start", json={"command": "ls -l"})
        response = response.json()
        print(response)
        #experiment()  # Start the experiment loop
    else:
        print("Experiment already running")

def check_experiment_thread() -> None:
    if st.session_state['experiment_running']:
        response = requests.get(f"{BASE_URL}/status")
        response = response.json()
        print(response)

        #If process ended, set experiment_running to false
        if response['status']!='running':
            st.session_state['experiment_running'] = False

    else:
        print("No experiment is running")

# Function to handle stopping the experiment
def stop_experiment_thread() -> None:
    if st.session_state['experiment_running']:
        response = requests.post(f"{BASE_URL}/stop")
        response = response.json()
        print(response)
    else:
        print("No experiment is running")

    st.session_state['experiment_running'] = False


# Sidebar buttons to control experiment start/stop
st.sidebar.button("Start Experiment", on_click=start_experiment_thread)
st.sidebar.button("Check Experiment", on_click=check_experiment_thread)
st.sidebar.button("Stop Experiment", on_click=stop_experiment_thread)


# Poll for new data and update the plots
def update_plots() -> None:
    # Plot 1
    fig1, ax1 = plt.subplots()
    ax1.plot(st.session_state['data1'])
    ax1.set_title("Plot 1")

    # Plot 2
    fig2, ax2 = plt.subplots()
    ax2.plot(st.session_state['data2'])
    ax2.set_title("Plot 2")

    # Display the plots in placeholders
    placeholder1.pyplot(fig1)
    placeholder2.pyplot(fig2)


# Continuously update the plots whether experiment is running or not
update_plots()


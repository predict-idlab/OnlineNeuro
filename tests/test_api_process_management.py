import subprocess
import time

import psutil
import pytest
import requests

# Configuration
API_PORT = 10002  # Use a different port for testing
BASE_URL = f"http://localhost:{API_PORT}"
MOCK_EXPERIMENT_PATH = "tests/test_experiment.py"


@pytest.fixture(scope="module")
def api_server():
    """Starts the Flask API in a background process for the test module."""
    command = ["python3", "api/app.py", "--port", str(API_PORT)]
    server_process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    # Wait for the server to start
    time.sleep(3)

    try:
        response = requests.get(f"{BASE_URL}/status", timeout=2)
        assert response.status_code == 200
    except requests.ConnectionError:
        server_process.kill()
        stdout, stderr = server_process.communicate()
        pytest.fail(
            f"Flask server failed to start on port {API_PORT}.\n"
            f"STDOUT: {stdout.decode()}\nSTDERR: {stderr.decode()}"
        )

    print(f"Flask server started with PID: {server_process.pid}")
    yield server_process  # Yield the process object to access its PID

    print(f"Shutting down Flask server with PID: {server_process.pid}")
    server_process.terminate()
    try:
        server_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server_process.kill()


def find_child_process_by_name(parent_pid, name):
    """Utility to find a descendant process of the API server by name."""
    try:
        parent = psutil.Process(parent_pid)
        # Search recursively for all children and grandchildren
        descendants = parent.children(recursive=True)
        for process in descendants:
            # Check if the mock script path is in the command line arguments
            if name in " ".join(process.cmdline()):
                return process
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None
    return None


def test_start_and_stop_normal_experiment(api_server):
    """
    Tests the full API flow: /start a normal process, check it's running, then /stop it.
    """
    # 1. Start the 'normal' experiment via the API
    payload = {"test_command": ["python3", MOCK_EXPERIMENT_PATH, "normal"]}
    start_response = requests.post(f"{BASE_URL}/start", json=payload)
    assert start_response.status_code == 200
    assert start_response.json()["status"] == "started"
    time.sleep(1)  # Give the process time to spawn

    # 2. Verify the child process is running
    child_proc = find_child_process_by_name(api_server.pid, MOCK_EXPERIMENT_PATH)
    assert child_proc is not None and child_proc.is_running()

    # 3. Stop the experiment via the API
    stop_response = requests.post(f"{BASE_URL}/stop")
    assert stop_response.status_code == 200
    assert stop_response.json()["status"] in ["stopped", "Forcefully stopped"]
    time.sleep(1)  # Give time for the process to be terminated

    # 4. Verify the child process is no longer running
    assert not child_proc.is_running()


def test_api_stops_stubborn_process(api_server):
    """
    Tests that the API's timeout/kill logic correctly handles a stubborn process.
    """
    # 1. Start the 'stubborn' experiment
    payload = {"test_command": ["python3", MOCK_EXPERIMENT_PATH, "stubborn"]}
    requests.post(f"{BASE_URL}/start", json=payload)
    time.sleep(1)

    # 2. Verify it's running
    child_proc = find_child_process_by_name(api_server.pid, MOCK_EXPERIMENT_PATH)
    assert child_proc is not None and child_proc.is_running()

    # 3. Stop it. The API should handle the timeout and force-kill internally.
    stop_response = requests.post(f"{BASE_URL}/stop")
    assert stop_response.status_code == 200
    # Your improved /stop endpoint should return 'Forcefully stopped' in this case
    assert stop_response.json()["status"] == "Forcefully stopped"

    # 4. Verify the process is gone
    time.sleep(1)  # Give it a moment to disappear from the process table
    assert not psutil.pid_exists(child_proc.pid)


def test_api_stops_spawner_process_and_grandchild(api_server):
    """
    Tests that stopping a 'spawner' via the API also cleans up its grandchild process.
    """
    # 1. Start the 'spawner' experiment
    payload = {"test_command": ["python3", MOCK_EXPERIMENT_PATH, "spawner"]}
    requests.post(f"{BASE_URL}/start", json=payload)
    time.sleep(2)  # Give it time to spawn the grandchild

    # 2. Verify both parent and grandchild are running
    child_proc = find_child_process_by_name(api_server.pid, MOCK_EXPERIMENT_PATH)
    assert child_proc is not None and child_proc.is_running()

    grandchild_proc = find_child_process_by_name(api_server.pid, "sleep")
    assert grandchild_proc is not None and grandchild_proc.is_running()

    # 3. Stop the experiment via the API
    stop_response = requests.post(f"{BASE_URL}/stop")
    assert stop_response.status_code == 200
    assert stop_response.json()["status"] in ["stopped", "Forcefully stopped"]

    # 4. Verify both processes are gone
    time.sleep(1)
    assert not psutil.pid_exists(child_proc.pid)
    assert not psutil.pid_exists(grandchild_proc.pid)

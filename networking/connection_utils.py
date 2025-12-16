# networking/connection_utils.py
import atexit
import socket
import threading
import warnings
from pathlib import Path
from threading import Thread

from online_neuro.utils.execution_utils import run_matlab, run_python_script

from .tcp_protocol import receive_message, send_message


def setup_networking(
    connection_config: dict, problem_config: dict, verbose: bool = True
) -> tuple[socket.socket, socket.socket]:
    """
    Establishes the TCP socket connection between the Python controller (server)
    and the external simulator (client/target).

    This function coordinates the initiation of the connection (creating the server
    and accepting the client), waits for the required simulator handshake, and
    registers a cleanup routine to ensure sockets are properly closed upon exit.

    Parameters
    ----------
    connection_config : dict
        Configuration dictionary containing networking parameters such as 'host'
        and 'port' for the TCP connection.
    problem_config : dict
        Configuration dictionary detailing the specific problem or simulation.
        This is passed to `start_connection` which passes it as first package to
        the client.
    verbose : bool, optional
        If True, prints status messages during the connection phase. Defaults to True.

    Returns
    -------
    tuple[socket.socket, socket.socket]
        A tuple containing:
        1. The server (listening) socket.
        2. The accepted client (connection) socket, used for data transfer.
    """
    server_socket, client_socket = start_connection(
        connection_config=connection_config, problem_config=problem_config
    )
    handshake_check(client_socket=client_socket, verbose=verbose)

    atexit.register(
        cleanup_server_socket, client_sock=client_socket, server_sock=server_socket
    )
    return server_socket, client_socket


def find_available_port(
    server_socket: socket.socket, ip: str, port: int
) -> tuple[socket.socket, int]:
    """
    Attempts to bind the given socket to the specified IP address and port.
    If the port is currently in use (raises OSError), it incrementally searches
    for the next available port until a successful binding occurs.

    This function modifies the provided `server_socket` by binding it to the
    first available address/port combination found.

    Parameters
    ----------
    server_socket : socket.socket
        The previously created TCP socket object (AF_INET, SOCK_STREAM) that needs to be bound.
    ip : str
        The IP address (e.g., '127.0.0.1' or 'localhost') to bind the socket to.
    port : int
        The initial port number to attempt binding to.

    Returns
    -------
    tuple[socket.socket, int]
        A tuple containing:
        1. The *bound* `server_socket` object.
        2. The integer port number that was successfully bound.

    Notes
    -----
    - The search for an available port starts at the requested `port` and increments by 1.
    - If a port is found, the loop terminates immediately, and the socket remains bound.
    """
    while True:
        try:
            server_socket.bind((ip, port))
            return server_socket, port

        except OSError:
            port += 1


def start_connection(
    connection_config: dict, problem_config: dict | None = None, verbose: bool = True
) -> tuple[socket.socket, socket.socket]:
    """
    Initializes the TCP server socket, binds it to a specified or available port,
    launches the external simulator process (MATLAB or Python) in a separate thread,
    and blocks until the simulator connects back to the server.

    Parameters
    ----------
    connection_config : dict
        Networking configuration including 'ip', 'port', and 'target'
        (specifying 'MATLAB' or 'Python' simulator).
    problem_config : dict | None, optional
        Configuration detailing the optimization problem, passed to the external
        simulator upon launch. Defaults to an empty dictionary if None.
    verbose : bool, optional
        If True, prints connection details and simulator launch payload. Defaults to True.

    Returns
    -------
    tuple[socket.socket, socket.socket]
        A tuple containing:
        1. The listening `server_socket`.
        2. The accepted `client_socket` connected to the simulator.

    Raises
    ------
    NotImplementedError
        If the 'target' specified in `connection_config` is neither 'MATLAB' nor 'Python'.
    RuntimeWarning
        If the desired port is unavailable and a new port is assigned.
    """
    if problem_config is None:
        problem_config = dict()

    # Create a TCP/IP socket
    server_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)

    if verbose:
        print(
            f"Establishing connection at port {connection_config['ip']} with port {connection_config['port']}"
        )
    # Bind the socket to the address and port
    server_socket, new_port = find_available_port(
        server_socket=server_socket,
        ip=connection_config["ip"],
        port=connection_config["port"],
    )

    if new_port != connection_config["port"]:
        msg = f"Selected port {connection_config['port']} not available, port {new_port} assigned instead."
        warnings.warn(msg)
        connection_config["port"] = new_port

    # Listen for incoming connections
    server_socket.listen(1)
    if verbose:
        print("Waiting for a connection...")

    payload = dict()
    payload["connection_config"] = connection_config.copy()
    payload["problem_config"] = problem_config.copy()

    if verbose:
        print(f"Starting {connection_config['target']} with Payload:")
        print(payload)

    target = connection_config["target"].lower()

    if target == "matlab":
        # Start process via Matlab engine and threading
        payload["script_path"] = str(Path("simulators") / "matlab")

        t = Thread(
            target=run_matlab,
            kwargs={
                "matlab_script_path": payload["script_path"],
                "matlab_function_name": "main",
                **payload,
            },
        )
        t.start()

    elif target == "python":
        # Start proces with thread
        payload["script_path"] = Path("simulators") / "python"
        process = run_python_script(function_name="main.py", **payload)

        def lock_process(proc):
            lock = threading.Lock()
            with lock:
                proc.wait()

        Thread(target=lock_process, args=(process,), daemon=True).start()

    else:
        msg = (
            f"No implementation available for simulator: {connection_config['target']}"
        )
        raise NotImplementedError(msg)

    # Accept a connection
    client_socket, client_address = server_socket.accept()

    return server_socket, client_socket


def handshake_check(
    client_socket: socket.socket, size_lim: int = 65536, verbose: bool = False
) -> None:
    """
    Performs a bidirectional handshake check with the connected simulator (client).

    The process involves:
    1. Receiving an initial handshake message (containing a 'dummyNumber') from the client.
    2. Validating the received message format.
    3. Constructing and sending a response message back to the client, echoing the 'dummyNumber'.
    4. Validating the transmission of the response message.

    Parameters
    ----------
    client_socket : socket.socket
        The connected client socket used for communication.
    size_lim : int, optional
        Maximum buffer size per package. This parameter might be primarily used
        by the underlying `receive_message` function. Defaults to 65536.
    verbose : bool, optional
        If True, prints detailed steps and the content of the messages exchanged.
        Defaults to False.

    Returns
    -------
    None

    Raises
    ------
    ConnectionError
        If the initial handshake request is not received or if the handshake
        response fails to be successfully sent back to the client.
    ValueError
        If the received handshake message is not a dictionary or does not contain
        the required "dummyNumber" key.
    Exception
        Propagates any underlying networking or processing exceptions.
    """
    try:
        if verbose:
            print("Server: Waiting for client handshake...")

        handshake_request = receive_message(client_socket, timeout=60)

        if handshake_request is None:
            raise ConnectionError("Failed to receive handshake request from client.")

        if verbose:
            print(f"Received handshake request: {handshake_request}")

        if (
            not isinstance(handshake_request, dict)
            or "dummyNumber" not in handshake_request
        ):
            raise ValueError(f"Invalid handshake request format: {handshake_request}")

        handshake_response = {
            "message": "Hello from Optimizer, handshake successful",
            "dummyNumber": handshake_request.get("dummyNumber"),  # Echo the number back
        }

        if verbose:
            print(" SERVER: Sending handshake back ...")

        result = send_message(client_socket, handshake_response)

        if verbose:
            print("Result received: ")
            print(f"       {result}")

        if result.get("status") != "success":
            raise ConnectionError("Failed to send handshake response to simulator.")

        if verbose:
            print(" Handshake complete. Starting main logic...")
            print("Received data:", handshake_request)
            print("Data sent back", handshake_response)

    except Exception as exception:
        print(f"An error occurred: {exception}")
        raise
    return None


def cleanup_server_socket(
    client_sock: socket.socket, server_sock: socket.socket, verbose: bool = False
) -> None:
    """
    Safely attempts to close both the accepted client connection socket and
    the listening server socket.

    This function is primarily intended to be registered with `atexit.register`
    to ensure resources are released when the experiment runner terminates.

    Parameters
    ----------
    client_sock : socket.socket
        The accepted connection socket communicating with the external simulator.
    server_sock : socket.socket
        The listening socket that accepted the connection.
    verbose : bool, optional
        If True, prints status messages during the cleanup process. Defaults to False.

    Returns
    -------
    None
    """
    if verbose:
        print("Cleaning up server socket...")
    try:
        client_sock.close()
        server_sock.close()
        if verbose:
            print("Server socket closed.")
    except Exception as exception:
        print(f"Error closing server socket: {exception}")

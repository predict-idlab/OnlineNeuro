import json
import socket
import threading
import warnings
from pathlib import Path
from threading import Thread

from online_neuro.utils import run_matlab, run_python_script


def find_available_port(server_socket: socket.socket,
                        ip: str,
                        port: int) -> tuple[socket.socket, int]:
    """
    Create a connection in an available port iterating until socket is bind.

    @param server_socket
    @param ip: ip address
    @param port:
    @return:
    """
    while True:
        try:
            server_socket.bind((ip, port))
            return server_socket, port

        except OSError:
            port += 1


def start_connection(connection_config: dict, problem_config: dict|None = None, verbose=True):
    """
    @param connection_config:
    @param problem_config:
    @param verbose:
    @return:
    """
    if problem_config is None:
        problem_config = dict()
    # Create a TCP/IP socket
    server_socket = socket.socket(family=socket.AF_INET,
                                  type=socket.SOCK_STREAM)

    if verbose:
        print(f"Establishing connection at port {connection_config['ip']} with port {connection_config['port']}")
    # Bind the socket to the address and port
    server_socket, new_port = find_available_port(server_socket=server_socket,
                                                  ip=connection_config['ip'],
                                                  port=connection_config['port'])

    if new_port != connection_config['port']:
        msg = f"Selected port {connection_config['port']} not available, port {new_port} assigned instead."
        warnings.warn(msg)
        connection_config['port'] = new_port

    # Listen for incoming connections
    server_socket.listen(1)
    if verbose:
        print('Waiting for a connection...')

    payload = dict()
    payload['connection_config'] = connection_config.copy()
    if 'experiment' in problem_config:
        payload['problem_name'] = problem_config['experiment'].get('name')
        payload['problem_type'] = problem_config['experiment'].get('type')

    omit_keys = ['experiment']
    other_keys = [k for k in problem_config.keys() if k not in omit_keys]
    if len(other_keys) > 0:
        payload['problem_config'] = {k: problem_config[k] for k in other_keys}
    else:
        raise Exception('Problem configuration contains no features to optimize.')

    if verbose:
        print(f"Starting {connection_config['target']} with Payload:")
        print(payload)

    if connection_config['target'] == 'MATLAB':
        # Start process via Matlab engine and threading
        payload['script_path'] = str(Path('simulators') / 'matlab')

        t = Thread(target=run_matlab, kwargs={'matlab_script_path': payload['script_path'],
                                              'matlab_function_name': 'main',
                                              **payload
                                              })
        t.start()

    elif connection_config['target'] == 'Python':
        # Start proces with thread
        # TODO, verify if the lock_process is still needed
        payload['script_path'] = Path('simulators') / 'python'
        process = run_python_script(function_name='main.py',
                                    **payload)

        def lock_process(proc):
            lock = threading.Lock()
            with lock:
                proc.wait()

        Thread(target=lock_process, args=(process,), daemon=True).start()

    else:
        msg = f"No implementation available for simulator: {connection_config['target']}"
        raise NotImplementedError(msg)

    # Accept a connection
    client_socket, client_address = server_socket.accept()

    return server_socket, client_socket


def handshake_check(client_socket, size_lim: int = 65536, verbose: bool = False):
    """
    @param client_socket
    @param size_lim per package
    @param verbose
    @return:
    """
    try:
        data = client_socket.recv(size_lim).decode()
        data = json.loads(data)

        response = {'message': 'Hello from Python',
                    'randomNumber': data['dummyNumber']}

        response_json = json.dumps(response) + '\n'
        client_socket.sendall(response_json.encode())
        if verbose:
            print('Doing a handshake...')
            print('Received data:', data)
            print('Data sent back', response_json.encode())

    except Exception as exception:
        print(f'An error occurred: {exception}')
        raise
    return None


def cleanup_server_socket(client_sock, server_sock, verbose: bool = False):
    if verbose:
        print('Cleaning up server socket...')
    try:
        client_sock.close()
        server_sock.close()
        if verbose:
            print('Server socket closed.')
    except Exception as exception:
        print(f'Error closing server socket: {exception}')

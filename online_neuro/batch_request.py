# online_neuro/batch_requests.py
import socket
import json
import sys
import pandas as pd
import numpy as np

import warnings
import time
from threading import Thread
from pathlib import Path
from utils import run_matlab, generate_grids, fetch_data
from common.utils import load_json
NUM_POINTS = 3


# TODO time this for the evaluation
def main(matlab_call=True, *args, **kwargs) -> None:
    # Load port configuration
    print(f"matlab_call flag: {matlab_call}")
    config_file = Path(__file__).parent / "config.json"
    config = load_json(config_file)

    # Create a TCP/IP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"Establishing connection at port {config['ip']} with port {config['port']}")
    # Bind the socket to the address and port
    server_socket.bind((config['ip'], config['port']))

    # Listen for incoming connections
    server_socket.listen(1)
    print("Waiting for a connection...")

    # Start Matlab process via engine and threading
    if matlab_call:
        t = Thread(target=run_matlab, kwargs={"matlab_initiate": False})
        t.start()

    # Accept a connection
    client_socket, client_address = server_socket.accept()
    print("Connection from:", client_address)

    # Receive data
    data = client_socket.recv(1024).decode()
    data = json.loads(data)
    print("Received data:", data)

    # Respond back
    response = {"message": "Hello from Python",
                "randomNumber": data['dummyNumber']}

    response_json = json.dumps(response) + "\n"
    client_socket.sendall(response_json.encode())
    print("Data sent back to matlab", response_json.encode())

    received_data = client_socket.recv(1024).decode()
    exp_config = json.loads(received_data)

    print("Received from MATLAB:")
    print(exp_config)
    features = exp_config['n_features']
    num_features = len(features)
    train_grid, test_grid = generate_grids(num_features, NUM_POINTS)

    lower_bound = np.array(exp_config['lower_bound'])
    upper_bound = np.array(exp_config['upper_bound'])

    train_grid = lower_bound + train_grid*(upper_bound - lower_bound)
    test_grid = lower_bound + test_grid*(upper_bound - lower_bound)

    full_grid = np.concatenate([train_grid, test_grid])
    print(f"Test points: {full_grid.shape}")

    response = {'message': 'All data points to query',
                'query_points': full_grid.tolist(),
                'terminate_flag': True}

    response_json = json.dumps(response) + "\n"
    client_socket.sendall(response_json.encode())

    received_data = fetch_data(client_socket)
    received_data = pd.DataFrame(received_data)

    save_df = pd.DataFrame(full_grid, columns=exp_config['n_features'])
    if len(save_df) != len(received_data):
        msg = f"Features and results with different lengths {received_data.shape}, {save_df.shape}"
        warnings.warn(msg)

    for col in received_data.columns:
        if col not in save_df:
            save_df[col] = received_data[col]
        else:
            warnings.warn(f"Column {col} already present in array (verify!)")

    train_df = save_df.iloc[:len(train_grid), :]
    test_df = save_df.iloc[len(train_grid):, :]

    exp_config_file = Path("config_experiments") / f"experiment_{exp_config['name']}.json"
    carried_exp = load_json(exp_config_file)
    default_config_file = Path("config_experiments") / f"parameter_spec_{exp_config['name']}.json"
    default_exp = load_json(default_config_file)

    for k, v in default_exp.items():
        if k not in carried_exp:
            carried_exp[k] = default_exp

    diam = carried_exp['dia']['value']
    dip_dis = abs(carried_exp['e_pos']['value'][0] - carried_exp['e_pos']['value'][2])

    save_path = Path(f"{config['benchmark_path']}") / f"{exp_config['name']}" / f"fib_dia_{diam}mm_dip_dis{dip_dis}mm_{NUM_POINTS}/"
    save_path.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(save_path / "train_df.csv", index=False)
    test_df.to_csv(save_path / "test_df.csv", index=False)

    with open(save_path / "metadata.json", 'w') as output_file:
        json.dump(carried_exp, output_file, indent=4)
    print(f"Results saved to {save_path}")

    # Close the connection
    client_socket.close()
    server_socket.close()


if __name__ == "__main__":
    parsed_args = {}
    for arg in sys.argv[1:]:
        if '=' in arg:
            key, value = arg.split('=', 1)
            parsed_args[key] = value
        else:
            parsed_args[arg] = True

    main(**parsed_args)

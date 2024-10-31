import json
import socket
import os
import random
from problems import circle, rosenbrock, log_single_var, toy_feasbility, vlmop2

problem_dict = {
    "circle_classification": circle,
    "rose_regression": rosenbrock,
    "multiobjective_problem": vlmop2,
}


def read_data(client_socket):
    # Assuming we are reading a fixed-size buffer, you may want to change this based on actual implementation
    received_data = client_socket.recv(1024).decode('utf-8')
    return json.loads(received_data)


def write_data(client_socket, data):
    json_data = json.dumps(data).encode('utf-8')
    client_socket.sendall(json_data)


def fun_wrapper(eval_fun, qp, eval_dict=None, mode=None):
    # Placeholder for your function logic; you'll need to implement it
    # This function can vary based on your actual problem configuration
    pass


def main(json_data):
    print(json_data)

    root_path = os.path.abspath(os.path.join(os.getcwd(), '../../'))

    data = json.loads(json_data)
    print(data['problem_config'])

    problem_name = data['problem_name']
    problem_type = data['problem_type']
    connection_config = data['connection_config']

    # Detect if path or configuration was provided
    problem_config = data['problem_config']
    if isinstance(problem_config, str):
        file_path = os.path.join(root_path, problem_config)
        with open(file_path) as f:
            problem_config = json.load(f)

    print("Verifying port is open")
    # Open TCP connection
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as tcpip_client:
        tcpip_client.connect((connection_config['ip'], connection_config['port']))

        # Send handshake
        data_to_send = {
            'message': 'Hello from Python',
            'dummyNumber': 123
        }
        write_data(tcpip_client, data_to_send)
        print("Data sent")

        # Wait for data
        print('Waiting for data...')
        received_data = read_data(tcpip_client)
        print(received_data)
        # End handshake, connection works.

    eval_fun = problem_dict[problem_name]
    # TODO
    eval_dict = dict()

    exp_summary = {
        'features': eval_dict,
        'constraints': ''
    }

    write_data(tcpip_client, exp_summary)
    print("Data sent")
    received_data = read_data(tcpip_client)
    query_points = received_data['query_points']

    num_points = len(query_points)
    fvalues = []

    for i in range(num_points):
        qp = query_points[i]
        result = fun_wrapper(eval_fun, qp, eval_dict, problem_name)
        fvalues.append(result)

    data_to_send = {'init_response': fvalues}
    json_data = json.dumps(data_to_send)
    size_lim = 1024

    if len(json_data) > size_lim:
        print("Breaking down message")
        pckgs = len(json_data) // size_lim + 1
        row_per_pckg = len(fvalues) // pckgs

        for i in range(pckgs):
            start_ix = i * row_per_pckg
            end_ix = min((i + 1) * row_per_pckg, len(fvalues))
            chunk = {
                'data': fvalues[start_ix:end_ix],
                'tot_pckgs': pckgs
            }
            write_data(tcpip_client, chunk)
    else:
        write_data(tcpip_client, data_to_send)

    print("Beginning loop now")
    terminate_flag = False

    while not terminate_flag:
        received_data = read_data(tcpip_client)
        qp = received_data['query_points']

        num_points = len(qp)
        terminate_flag = received_data.get('terminate_flag', False)

        if terminate_flag:
            print("Termination signal received from Python. Saving data and closing connection...")
        else:
            print("Requested query points")
            fvalues = [fun_wrapper(eval_fun, qp[i], eval_dict, problem_name) for i in range(num_points)]
            send_data = {'observations': fvalues}
            write_data(tcpip_client, send_data)

    tcpip_client.close()


# Run this in main script if necessary:
if __name__ == "__main__":
    main()
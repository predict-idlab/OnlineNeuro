# simulators/python/main.py
import argparse
import atexit
import json
import select
import socket
import time
from collections import defaultdict

from problems import circle, rosenbrock, vlmop2

try:
    from problems import cajal_problems
except ImportError as e:
    # TODO this warning needs to be passed to the GUI in case running over it.
    msg = "Cajal is not installed, Cajal problems are not loaded."
    raise ImportError(msg) from e

problem_dict = {
    "circle_classification": circle,
    "rose_regression": rosenbrock,
    "moo_problem": vlmop2,
    "cajal_ap_block": cajal_problems.cajal_fun,
}

VECTORIZED_FUNS = [circle, rosenbrock, vlmop2]


def send_data(fvalues, tcpip_client, size_lim=65536):
    """
    TODO current function is meant to only send the output.
         This needs to be changed in case we want to send additional data (dict with multiple entries)

    @param fvalues:
    @param tcpip_client:
    @param size_lim:
    @return:
    """

    json_data = json.dumps(fvalues)
    print("Data to be sent")
    print(json_data)

    if len(json_data) > size_lim:
        json_data = json.dumps(fvalues[0])

        if len(json_data) < size_lim:
            pckgs = len(fvalues)
            for i in range(pckgs):
                print(f"Package {i+1}/{pckgs}", end="\n")
                chunk = {
                    "data": fvalues[i],
                    "tot_pckgs": pckgs,
                    "current_pckg": i,  # Track which chunk this is
                }
                response = write_data(tcpip_client, chunk)
        else:
            raise NotImplementedError(
                "Package needs to be transmitted in another way (too large for socket)"
            )

    else:
        response = write_data(tcpip_client, fvalues)

    return response


def convert_list_to_dict(list_of_dicts):
    # Convert to dict of lists
    dict_of_lists = defaultdict(list)

    for d in list_of_dicts:
        for key, value in d.items():
            dict_of_lists[key].append(value)

    return dict(dict_of_lists)


def evaluate_function(query_points, fixed_features, eval_fun, problem_name, mode):
    if eval_fun in VECTORIZED_FUNS:
        qp_dict = convert_list_to_dict(query_points)
        for key, value in fixed_features.items():
            if key in qp_dict:
                raise ValueError(
                    f"Key '{key}' from fixed_features already exists in query_points."
                )
            qp_dict[key] = [value] * len(query_points)

        fvalues = fun_wrapper(
            eval_fun, qp=qp_dict, problem_name=problem_name, mode=mode
        )

    else:
        num_points = len(query_points)
        fvalues = []
        for i in range(num_points):
            print(f"Exp {i}/{num_points}")
            qp = query_points[i]
            qp_with_fixed = {**qp, **fixed_features}
            result = fun_wrapper(
                eval_fun, qp=qp_with_fixed, problem_name=problem_name, mode=mode
            )
            fvalues.append(result)

    return fvalues


def read_data(client_socket, timeout=300):
    """
    @param client_socket:
    @param timeout:
    @return:
    """
    buffer = b""
    end_time = time.time() + timeout

    while time.time() < end_time:
        ready_to_read, _, _ = select.select([client_socket], [], [], 1)
        if ready_to_read:
            part = client_socket.recv(1024)
            if part:
                buffer += part

            # Check if buffer contains a complete JSON message ending with newline
            if b"\n" in buffer:
                messages = buffer.split(b"\n")

                # Process each complete message
                for message in messages[:-1]:  # Ignore any trailing partial message
                    if message:
                        try:
                            decoded_message = json.loads(
                                message.decode("utf-8").strip()
                            )
                            print("Decoded message:", decoded_message)
                            # Update buffer before returning
                            buffer = messages[-1] if messages[-1] else b""
                            return decoded_message
                        except json.JSONDecodeError as e:
                            print("JSON decoding error:", e, "Message:", message)

                # Retain last part as buffer in case it's a partial message
                buffer = messages[-1]

    raise TimeoutError("Timeout: No complete JSON message received.")


def write_data(client_socket, data) -> dict:
    json_data = json.dumps(data, ensure_ascii=False).encode("utf-8")
    try:
        client_socket.sendall(json_data)
        return {"Message": "Data sent to server"}
    except Exception as e:
        msg = e
        return {"Message": f"Something went wrong {msg}"}


def fun_wrapper(eval_fun, qp, problem_name, mode=None):
    """
    Logic on how to post-process the results of the evaluated function
    @param eval_fun:
    @param qp:
    @param problem_name:
    @param mode:
    @return:
    """
    if problem_name == "placeholder":
        return 0
    else:
        return eval_fun(**qp)


def main(problem_config, connection_config, *args, **kwargs):
    # Detect if path or configuration was provided
    problem_name = problem_config["experiment_parameters"]["problem_name"]

    tcpip_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcpip_client.settimeout(connection_config["Timeout"])
    ## buffer_size = connection_config['SizeLimit']

    atexit.register(tcpip_client.close)

    tcpip_client.connect((connection_config["ip"], connection_config["port"]))
    tcpip_client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    # Send handshake
    data_to_send = {"message": "Hello from Python", "dummyNumber": 123}
    write_data(tcpip_client, data_to_send)
    received_data = read_data(tcpip_client)

    eval_fun = None
    if problem_name in problem_dict:
        eval_fun = problem_dict[problem_name]
    else:
        NotImplementedError(f"Problem {problem_name} not implemented.")

    print(problem_name)
    print(problem_dict)
    received_data = read_data(tcpip_client)
    fixed_features = received_data["Fixed_features"]

    received_data = read_data(tcpip_client)
    query_points = received_data["query_points"]

    operator = None
    # TODO some calls need to be made point by point, but when possible, is best to use the vector form

    fvalues = evaluate_function(
        query_points=query_points,
        fixed_features=fixed_features,
        eval_fun=eval_fun,
        problem_name=problem_name,
        mode=operator,
    )

    response = send_data(fvalues=fvalues, tcpip_client=tcpip_client)

    print(response)
    print("Main side: Entering loop now")
    terminate_flag = False

    while not terminate_flag:
        received_data = read_data(tcpip_client)
        query_points = received_data["query_points"]
        terminate_flag = received_data.get("terminate_flag", False)

        if terminate_flag:
            print(
                "Termination signal received from Python. Saving data and closing connection..."
            )
        else:
            fvalues = evaluate_function(
                query_points=query_points,
                fixed_features=fixed_features,
                eval_fun=eval_fun,
                problem_name=problem_name,
                mode=operator,
            )

            response = send_data(fvalues=fvalues, tcpip_client=tcpip_client)

            print(f"response {response}")

    tcpip_client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for handling complex arguments."
    )

    # Define arguments
    parser.add_argument(
        "--connection_config",
        type=str,
        required=True,
        help="Path or settings for the connection configuration",
    )
    # parser.add_argument(
    #     '--problem_name', type=str, required=True, help='Name of the problem'
    # )
    # parser.add_argument(
    #     '--problem_type', type=str, required=True, help='Type of the problem'
    # )
    parser.add_argument(
        "--problem_config",
        type=str,
        required=True,
        help="JSON string representing the problem configuration",
    )
    args = parser.parse_args()

    # Convert problem_config from JSON string to Python dictionary
    try:
        problem_config = json.loads(args.problem_config)
    except json.JSONDecodeError:
        print("Error: problem_config must be a valid JSON string.")
        exit(1)
    # Convert problem_config from JSON string to Python dictionary
    try:
        connection_config = json.loads(args.connection_config)
    except json.JSONDecodeError:
        print("Error: connection_config must be a valid JSON string.")
        exit(1)

    main(
        problem_config=problem_config,
        connection_config=connection_config,
    )

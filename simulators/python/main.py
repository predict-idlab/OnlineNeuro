# simulators/python/main.py
"""
TCP-based Python simulator entry point.

This module connects to a TCP server (typically a GUI/controller), receives
query points for a configured optimization/problem, evaluates the selected
problem function, and returns the results. It supports both vectorized and
point-wise evaluation depending on the problem implementation.

Notes
-----
- Configuration is passed as JSON strings via the command line.
- Some problems support vectorized evaluation for efficiency.
"""

import argparse
import atexit
import json
import socket
from collections import defaultdict
from typing import Any, Callable

from problems import circle, rosenbrock, vlmop2

from networking.tcp_protocol import receive_message, send_message

try:
    from problems import cajal_problems
except ImportError as e:
    # TODO this warning needs to be passed to the GUI in case running over it.
    msg = "Cajal is not installed, Cajal problems are not loaded."
    raise ImportError(msg) from e

ProblemFn = Callable[..., Any]
problem_dict: dict[str, ProblemFn] = {
    "circle_classification": circle,
    "python_rose_regression": rosenbrock,
    "moo_problem": vlmop2,
    "cajal_ap_block": cajal_problems.cajal_fun,
}

VECTORIZED_FUNS: list[ProblemFn] = [circle, rosenbrock, vlmop2]


def convert_list_to_dict(list_of_dicts: list[dict[str, Any]]) -> dict[str, list]:
    """
    Convert a list of dictionaries into a dictionary of lists.

    Parameters
    ----------
    list_of_dicts : Iterable[Mapping[str, Any]]
        Iterable where each element is a dictionary with identical keys.

    Returns
    -------
    list[dict[str, Any]]
        Dictionary mapping each key to a list of values collected from all
        dictionaries.
    """
    dict_of_lists = defaultdict(list)

    for d in list_of_dicts:
        for key, value in d.items():
            dict_of_lists[key].append(value)

    return dict(dict_of_lists)


def evaluate_function(
    query_points: list[dict],
    fixed_features: dict,
    eval_fun: ProblemFn,
    problem_name: str,
    mode: str | None = None,
    verbose: bool = False,
) -> list[Any]:
    """
    Evaluate a problem function on a set of query points.

    The function automatically selects vectorized or point-wise evaluation
    depending on whether the problem function is listed in ``VECTORIZED_FUNS``.

    Parameters
    ----------
    query_points : list[dict[str, Any]]
        List of query points to evaluate.
    fixed_features : Mapping[str, Any]
        Fixed parameters shared across all evaluations.
    eval_fun : Callable
        Problem evaluation function.
    problem_name : str
        Name of the problem.
    mode : str, optional
        Optional execution mode.
    verbose: bool
        Boolean to print the iteration number (for non vectorized functions)

    Returns
    -------
    list[Any]
        Evaluation results for each query point.
    """

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
            if verbose:
                print(f"Exp {i}/{num_points}")
            qp = query_points[i]
            qp_with_fixed = {**qp, **fixed_features}
            result = fun_wrapper(
                eval_fun, qp=qp_with_fixed, problem_name=problem_name, mode=mode
            )
            fvalues.append(result)

    return fvalues


def fun_wrapper(
    eval_fun: Callable, qp: dict, problem_name: str, mode: str | None = None
) -> Any:
    """
    Wrap a problem evaluation call to allow for special-case handling.

    Parameters
    ----------
    eval_fun : Callable
        Problem evaluation function.
    qp : Mapping[str, Any]
        Query point parameters passed as keyword arguments.
    problem_name : str
        Name of the problem being evaluated.
    mode : str, optional
        Optional execution mode (reserved for future extensions).

    Returns
    -------
    Any
        Result of the problem evaluation.
    """
    return eval_fun(**qp)


def main(problem_config, connection_config, *args, **kwargs):
    """
    Main entry point for the Python simulator process.


    Parameters
    ----------
    problem_config : Mapping[str, Any]
        Problem configuration dictionary.
    connection_config : Mapping[str, Any]
        TCP connection configuration dictionary.
    """

    # Detect if path or configuration was provided
    problem_name = problem_config["experiment_parameters"]["problem_name"]

    tcpip_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcpip_client.settimeout(connection_config["Timeout"])

    atexit.register(tcpip_client.close)

    tcpip_client.connect((connection_config["ip"], connection_config["port"]))
    tcpip_client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    # Send handshake
    data_to_send = {"message": "Hello from Python", "dummyNumber": 123}
    response = send_message(tcpip_client, data_to_send)
    received_data = receive_message(tcpip_client)
    print(" Main, received handshake:", received_data)

    response = send_message(
        tcpip_client, {"message": "Confirmed first message", "status": "ready"}
    )

    eval_fun = problem_dict.get(problem_name, None)
    if eval_fun is None:
        tcpip_client.close()
        raise NotImplementedError(f"Problem {problem_name} not implemented.")

    received_data = receive_message(tcpip_client)
    assert isinstance(
        received_data, dict
    ), "Expected response in the form of a dictionary"

    fixed_features = received_data.get("Fixed_features", {})

    received_data = receive_message(tcpip_client)
    assert isinstance(
        received_data, dict
    ), "Expected response in the form of a dictionary"
    query_points = received_data.get("query_points", None)
    if query_points is None:
        tcpip_client.close()
        raise ValueError("Query points not received, terminating ...")

    assert isinstance(query_points, list), "Expected query points int he form of a list"
    fvalues = evaluate_function(
        query_points=query_points,
        fixed_features=fixed_features,
        eval_fun=eval_fun,
        problem_name=problem_name,
    )

    response = send_message(tcpip_client, fvalues)

    if response["status"] != "success":
        raise ConnectionError("Stopping, the server did not sent a success status")

    terminate_flag = False

    while not terminate_flag:
        received_data = receive_message(tcpip_client)
        query_points = received_data.get("query_points", [])
        terminate_flag = received_data.get("terminate_flag", False)

        if not query_points:
            print(
                f"No query points received from Python, instead received: {received_data} \n terminating ..."
            )
            break

        if terminate_flag:
            msg = received_data.get("message", None)
            print(f"Termination signal received from Python with message: {msg}")
            payload = {"status": "ok", "message": "process terminated"}
            response = send_message(tcpip_client, payload)

        else:
            fvalues = evaluate_function(
                query_points=query_points,
                fixed_features=fixed_features,
                eval_fun=eval_fun,
                problem_name=problem_name,
            )

            response = send_message(tcpip_client, fvalues)

    tcpip_client.close()


# CLI
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

    main(problem_config=problem_config, connection_config=connection_config)

import os
import argparse

import socket
import json
import sys
import pandas as pd
import numpy as np

import tensorflow as tf
import warnings
import time

import trieste
from trieste.data import Dataset
from online_neuro.bayessian_optimizer import BayesianOptimizer
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.acquisition.function import BayesianActiveLearningByDisagreement, \
    NegativePredictiveMean, PredictiveVariance, ExpectedHypervolumeImprovement
# https://github.com/secondmind-labs/trieste/blob/c6a039aa9ecf413c7bcb400ff565cd283c5a16f5/trieste/acquisition/function/__init__.py
from online_learning import build_model

# AugmentedExpectedImprovement
# ExpectedImprovement
# ProbabilityOfImprovement
# NegativeLowerConfidenceBound
# NegativePredictiveMean
# ProbabilityOfFeasibility

# FastConstraintsFeasibility
# ExpectedConstrainedImprovement
# MonteCarloExpectedImprovement
# MonteCarloAugmentedExpectedImprovement
# MonteCarloExpectedImprovement
# BatchMonteCarloExpectedImprovement
# BatchExpectedImprovement
# MultipleOptimismNegativeLowerConfidenceBound

# MakePositive
from trieste.acquisition.rule import OBJECTIVE
from threading import Thread

from plotting import update_plot
from online_neuro.utils import customMinMaxScaler, run_matlab, fetch_data
from common.utils import load_json


def start_connection(config, debug_matlab=False):
    """
    @param config:
    @param initiator:
    @param target:
    @return:
    """
    # Create a TCP/IP socket
    server_socket = socket.socket(family=socket.AF_INET,
                                  type=socket.SOCK_STREAM)
    print(f"Establishing connection at port {config['ip']} with port {config['port']}")
    # Bind the socket to the address and port
    server_socket.bind((config['ip'], config['port']))

    # Listen for incoming connections
    server_socket.listen(1)
    print("Waiting for a connection...")
    # Start Matlab process via engine and threading

    if config['initiator'] == "Python":
        print("Python initiates communication ....")
        if config['target'] == "MATLAB":
            if not debug_matlab:
                # Flag to allow for manually launching Matlab and debugging its end
                # If threading, Python launches the Matlab main, else, main needs to be manually launched
                matlab_payload = connection_config.copy()
                matlab_payload['problem'] = problem_config['name']
                matlab_payload['matlab_initiates'] =False
                matlab_payload['script_path'] = "problems/matlab/"

                t = Thread(target=run_matlab, kwargs={"matlab_script_path": matlab_payload['script_path'],
                                                      "matlab_function_name": "main",
                                                        **matlab_payload
                                                      })
                t.start()
            else:
                print("You are in debug mode, start Matlab manually")
        elif config['target'] == "Python":
            msg = f"No implementation available for target {config['target']}"
            raise NotImplementedError(msg)
        else:
            msg = f"No implementation available for target {config['initiator']}"
            raise NotImplementedError(msg)
    elif config['initiator'] == "MATLAB":
        print("Matlab initiated communication, starting Python logic now ...")
        if config['target'] != "MATLAB":
            msg = f"Target not valid ({config['target']}).\n If MATLAB initiates, it is because the controller is MATLAB."
            raise Exception(msg)
    else:
        msg = f"No implementation available for initiator {config['initiator']}"
        raise NotImplementedError(msg)

    # Accept a connection
    client_socket, client_address = server_socket.accept()
    print("Connection from:", client_address)

    return server_socket, client_socket


def handshake_check(client_socket):
    """
    @param client_socket:
    @return:
    """
    # Receive data
    try:
        data = client_socket.recv(1024).decode()
        data = json.loads(data)
        print("Received data:", data)

        # Respond back
        response = {"message": "Hello from Python",
                    "randomNumber": data['dummyNumber']}

        response_json = json.dumps(response) + "\n"
        client_socket.sendall(response_json.encode())
        print("Data sent back to matlab", response_json.encode())
    except Exception as e:
        print(f"An error occurred: {e}")
        raise
    return None

def define_search_space(scale_inputs:bool=True, problem_config:dict=None, gui_config:dict=None):
    """
    @param config: Global experiment configuration. More specifically, whether inputs need to be scaled.
            In most cases, input scaling should be used.
    @param problem_config: Information concerning the problem. Specifically, the upper and lower bounds.
    @param gui_config: Information concerning the search space as defined by the user.
                The gui boundaries should be within the problem_config range.
                If not the case, these are ignored.

    @return: Tuple (Trieste Search Space, scaler)

    """
    lower_bound = [problem_config['features'][k][0] for k in problem_config['features'].keys()]
    upper_bound = [problem_config['features'][k][1] for k in problem_config['features'].keys()]
    num_features = len(problem_config['features'])

    if scale_inputs:
        output_range = (-1, 1)

        lb = num_features * [output_range[0]]
        ub = num_features * [output_range[1]]

        search_space = trieste.space.Box(lower=lb,
                                         upper=ub)

        scaler = customMinMaxScaler(feature_min=lower_bound,
                                    feature_max=upper_bound,
                                    output_range=output_range)

    else:
        search_space = trieste.space.Box(lower=lower_bound,
                                         upper=upper_bound)
        scaler = None

    return search_space, scaler


def main(connection_config: dict, model_config: dict, path_config: dict, problem_config: dict, *args, **kwargs) -> None:
    default_dict = load_json("config.json")

    connection_config = {**default_dict['connection_config'], **connection_config}
    model_config = {**default_dict['model_config'], **model_config}
    path_config = {**default_dict['path_config'], **path_config}
    problem_config = {**default_dict['problem_config'], **problem_config}
    # TODO
    #  - verify here that the problem can be solved by the simulator (target)
    # i.e. matlab, axonsim, neuron, python, each need to have a simulator file for the given problem.
    #  - verify that the model type can solve the problem type
    # i.e. prevent multiobjective/regression to be solved with classification, etc...

    server_socket, client_socket = start_connection(config=connection_config,
                                                    debug_matlab=False)

    handshake_check(client_socket=client_socket)

    ## First batch of data
    # Passes 'name of fun', feature names, target_names, Upper boundary, and lower_boundary.
    # This should probably also pass constraints, although this needs to be thought still.

    received_data = client_socket.recv(1024).decode()
    problem_specs = json.loads(received_data)

    print("Received from MATLAB:")
    print(problem_specs)
    problem_config = {**problem_config, **problem_specs}

    search_space, scaler = define_search_space(scale_inputs=model_config["scale_inputs"],
                                               problem_config=problem_config)

    ## First batch requested to the simulator.
    # Initialize GP and searchs strategy with first batch of data.

    # TODO Optional. Add other sampling methods such as LHS (Halton is in Trieste but not needed).
    initial_qp = search_space.sample_sobol(num_samples=model_config['init_samples'])

    if scaler is not None:
        initial_qp_inv = scaler.inverse_transform(initial_qp)
    else:
        initial_qp_inv = initial_qp

    response = {'message': 'first queried points using Sobol method',
                'query_points': initial_qp_inv.tolist()}

    response_json = json.dumps(response) + "\n"
    client_socket.sendall(response_json.encode())

    received_data = fetch_data(client_socket)

    print("First data package:", received_data)
    received_data = pd.DataFrame(received_data)

    query_points = np.stack(initial_qp)
    observations = received_data['init_response'].values

    if isinstance(observations[0], list):
        observations = np.vstack(observations)

    init_data_size = len(query_points)

    if observations.ndim <= 1:
        observations = observations.reshape(-1, 1)

    feature_names = list(problem_config['features'].keys())
    results_df = pd.DataFrame(query_points, columns=feature_names)
    if observations.shape[1] > 1:
        for i in range(observations.shape[1]):
            results_df[f'response_{i}'] = observations[:, i]
    else:
        results_df['response'] = observations

    # TODO Complete file naming
    os.makedirs(f"{path_config['save_path']}/{problem_config['name']}/", exist_ok=True)
    results_df.to_csv(f"{path_config['save_path']}/{problem_config['name']}/results.csv", index=False)

    # Important note. For some tensorflow reason observations need to be float even for classification problems.
    # Additional note. Tf variance can only accept real numbers
    init_dataset = Dataset(query_points=tf.cast(query_points, tf.float64),
                           observations=tf.cast(observations, tf.float64))

    if model_config['noise_free']:
        model_config['kernel_variance'] = 1e-6
    else:
        model_config['kernel_variance'] = None

    model = build_model(init_dataset, search_space, model_config)

    # TODO extend to batch_sampling at a later stage
    if model_config['batch_sampling']:
        if 'num_query_points' in model_config:
            if model_config['num_query_points'] == 1:
                warnings.warn(f"Batch querying was specified, but config specifies only 1 sample")
        else:
            warnings.warn(f"Batch querying was specified, but config filed does not specify batch size, defaulting to three")
            model_config['num_query_points'] = 3

    if problem_config['type'] in ['multiobjective']:
        acq = ExpectedHypervolumeImprovement()
        print("Multiobjective model using HyperVolumes")
    elif problem_config['type'] in ['classification']:
        acq = BayesianActiveLearningByDisagreement()
        print("using Classification BALD")
    elif problem_config['type'] in ['regression']:
        if 'acquisition' in problem_config:
            if problem_config['acquisition'] == 'negative_predictive_mean':
                acq = NegativePredictiveMean()
            elif problem_config['acquisition'] == 'predictive_variance':
                acq = PredictiveVariance()
            else:
                msg = f"Acquisition [{problem_config['acquisition']}] not identified, defaulting to Predictive Variance (Global Search)"
                warnings.warn(msg)
                acq = PredictiveVariance()
        else:
            msg = f"No Acquisition specified in the config.json/ --flags. Defaulting to Predictive Variance (Global Search)"
            warnings.warn(msg)
            acq = PredictiveVariance()

        #TODO, extend acquisitions as required
        # Sample close to a given threshold
        # acq = ExpectedFeasibility(threshold=-0.5)
        # Equivalent to Maximizing a function
        # acq = NegativePredictiveMean()
        # Minimize global variance
        # acq = PredictiveVariance()

    rule = EfficientGlobalOptimization(builder=acq,
                                       num_query_points=model_config['num_query_points']
                                       )
    bo = BayesianOptimizer(observer=problem_config['name'],
                           search_space=search_space,
                           scaler=scaler,
                           track_state=True,
                           track_path=f"{path_config['save_path']}/{problem_config['name']}/",
                           acquisition_rule=rule)

    qp = bo.request_query_points(datasets=init_dataset,
                                 models=model,
                                 fit_initial_model=True,
                                 fit_model=True)
    terminate_flag = False
    if qp is None:
        raise Exception("Terminated before optimization started \n no query points were retrieved")

    qp_list = qp.tolist()

    with_plots = True
    count = 0

    # TODO send stop signals from both ends upon certain conditions (i.e. no improvement, reached goal,
    # buget limit, etc...)

    # TODO, for toy-problems, add the option of generating performance results (i.e. Accuracy/RMSE over
    # updates

    # TODO, for real-world problems, generate the GridSearch solution, store (within GIT?) And use to
    # generate example figures of online learning efficiency

    # TODO, plotting functions for Pareto

    # TODO, plotting functions for 3D search shapes (see Notebook in NeuoAAA repo)

    while not terminate_flag:
        # Counter is used to save figures with different names. If kept constant it overwrites the figure.
        count += 1

        # Check if termination signal received from MATLAB
        # Respond back
        message = {"query_points": qp_list,
                   "terminate_flag": terminate_flag}
        print(message)
        response_json = json.dumps(message) + "\n"
        client_socket.sendall(response_json.encode())

        if "terminate_flag" in received_data:
            terminate_flag = received_data['terminate_flag']
            print("Termination signal received from MATLAB \n Saving results... \n Closing connection...")

        received_data = client_socket.recv(1024).decode()
        received_data = json.loads(received_data)

        if with_plots:
            plot_data = bo.result.try_get_final_datasets()[OBJECTIVE]
            update_plot(bo,
                        initial_data=(plot_data.query_points[:init_data_size],
                                      plot_data.observations[:init_data_size]),
                        sampled_data=(plot_data.query_points[init_data_size:],
                                      plot_data.observations[init_data_size:]),
                        plot_ground_truth=None, ground_truth_function=None,
                        count=count)
            time.sleep(0.5)

        observations = received_data['observations']
        observations = np.array(observations)

        if isinstance(observations[0], list):
            observations = np.vstack(observations)
        if observations.ndim == 1:
            # TODO Need to fix this to allow batch sampling,
            #  but also multivariate modeling.
            if problem_config['type'] == 'multiobjective':
                print("Problem is MOB, reshaping...")
                observations = observations.reshape(-1, 2)
            else:
                observations = observations.reshape(-1, 1)
        print(observations.shape)

        bo.optimize_step(query_points=qp,
                         observer_output=observations)

        final_data = bo.result.try_get_final_datasets()[OBJECTIVE]
        final_models = bo.result.try_get_final_models()[OBJECTIVE]

        qp = bo.request_query_points(datasets=final_data,
                                     models=final_models,
                                     fit_initial_model=False,
                                     fit_model=True)

        qp_list = qp.tolist()

    # Close the connection
    client_socket.close()
    server_socket.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Controller to start Python or MATLAB process.")

    group1 = parser.add_mutually_exclusive_group(required=False)
    group2 = parser.add_mutually_exclusive_group(required=False)

    group1.add_argument('--config', type=str, help="Path to the global configuration file")
    group2.add_argument('--connection_config', type=str, help="Path to the connection's configuration file")
    group2.add_argument('--model_config', type=str, help="Path to the model's configuration file")
    group2.add_argument('--problem_config', type=str, help="Path to the problems's configuration file")
    group2.add_argument('--path_config', type=str, help="Path to the paths's configuration file")

    parser.add_argument('--initiator', choices=['Python', 'MATLAB'], default="Python",
                        required=False,
                        help="Specify the initiator process: 'Python' or 'MATLAB'")
    parser.add_argument('--target', choices=['Python', 'MATLAB'], default="MATLAB",
                        required=False,
                        help="Specify the target process to control: 'Python' or 'MATLAB'")

    # Unknown args allow for external calls from Streamlit or other frontends
    args, unknown_args = parser.parse_known_args()

    if args.config:

        config = load_json(args.config)
        connection_config = config.get('connection_config', {})
        model_config = config.get('model_config', {})
        path_config = config.get('path_config', {})
        problem_config = config.get('problem_config', {})

        other_params = dict()

    elif args.connection_config or args.model_config or \
            args.problem_config or args.path_config:

        connection_config = args.get('connection_config', {})
        model_config = args.get('model_config', {})
        path_config = args.get('path_config', {})
        problem_config = args.get('problem_config', {})

        other_params = dict()

    else:
        # Convert Namespace to dict
        args_dict = vars(args)
        print(f"Known args: {args_dict}")
        print(f"Unknown args: {unknown_args}")

        unknown_args_dict = {}
        i = 0
        while i < len(unknown_args):
            if unknown_args[i].startswith('--'):
                key = unknown_args[i][2:]  # Remove '--' from flag name
                # Check if the next element is another flag or if it's the last item in the list
                if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith('--'):
                    # Key-value pair
                    unknown_args_dict[key] = unknown_args[i + 1]
                    i += 2
                else:
                    # Standalone flag
                    unknown_args_dict[key] = True
            else:
                i += 1  # Skip if we encounter unexpected input (shouldn't happen)
        merged_args = {**unknown_args_dict, **args_dict}

        connection_config = dict()
        problem_config = dict()
        model_config = dict()
        path_config = dict()

        other_params = dict()
        for k, v in merged_args:
            try:
                key_split = k.split("_")
                k_beginning = key_split[0]
                actual_key = '_'.join(key_split[1:])
            except Exception as e:
                warnings.warn(f"{k} seems to not be a valid formatted flag, passing it directly to main")

            if k_beginning == 'conn':
                connection_config[actual_key] = v
            elif k_beginning == 'prob':
                problem_config[actual_key] = v
            elif k_beginning == 'path':
                path_config[actual_key] = v
            else:
                other_params[k] = v

    try:
        main(connection_config=connection_config,
             model_config=model_config,
             path_config=path_config,
             problem_config=problem_config,
             **other_params)

    except Exception as e:
        print(f"Error occurred: {e}")
        sys.exit(1)

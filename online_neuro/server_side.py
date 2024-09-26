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
from utils import customMinMaxScaler, run_matlab, fetch_data


def start_connection(config, initiator, target, debug_matlab=False):
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

    if initiator == "Python":
        print("Python initiates communication ....")
        if target == "MATLAB":

            if not debug_matlab:
                # Flag to allow for manually launching Matlab and debugging its end
                # If threading, Python launches the Matlab main, else, main needs to be manually launched
                t = Thread(target=run_matlab, kwargs={"matlab_initiate": False})
                t.start()
            else:
                print("You are in debug mode, start Matlab manually")
        elif target == "Python":
            msg = f"No implementation available for target {initiator}"
            raise NotImplementedError(msg)
        else:
            msg = f"No implementation available for target {initiator}"
            raise NotImplementedError(msg)
    elif initiator == "MATLAB":
        print("Matlab initiated communication, starting Python logic now ...")
        if target != "MATLAB":
            msg = f"Target not valid ({target}). If MATLAB initiates, it is because the controller is MATLAB."
            raise Exception(msg)
    else:
        msg = f"No implementation available for initiator {initiator}"
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
        print(f"An error ocurred: {e}")
        raise
    return None

def define_search_space(config, problem_config, gui_config=None):
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

    if config['experiment']['scale_inputs']:
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


def main(initiator="Python", target="MATLAB", *args, **kwargs) -> None:
    # Load port configuration
    print(f"Initiator flag: {initiator}")
    with open('../config.json', 'r') as f:
        config = json.load(f)

    # TODO verify here that the problem type matches model type
    # i.e. prevent multiobjective/regression to be solved with classification, etc...

    server_socket, client_socket = start_connection(config=config,
                                                    initiator=initiator,
                                                    target=target,
                                                    debug_matlab=False)

    handshake_check(client_socket=client_socket)

    ## First batch of data
    # Passes 'name of fun', feature names, target_names, Upper boundary, and lower_boundary.
    # This should probably also pass constraints, although this needs to be thought still.

    received_data = client_socket.recv(1024).decode()
    problem_config = json.loads(received_data)

    print("Received from MATLAB:")
    print(problem_config)

    search_space, scaler = define_search_space(config=config,
                                               problem_config=problem_config)

    ## First batch requested to the simulator.
    # Initialize GP and searchs strategy with first batch of data.

    # TODO Optional. Add other sampling methods such as LHS (Halton is in Trieste but not needed).
    initial_qp = search_space.sample_sobol(num_samples=config['experiment']['init_samples'])

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
    save_df = pd.DataFrame(query_points, columns=feature_names)
    if observations.shape[1] > 1:
        for i in range(observations.shape[1]):
            save_df[f'response_{i}'] = observations[:, i]
    else:
        save_df['response'] = observations

    # TODO Complete file naming
    os.makedirs(f"{config['save_path']}/{problem_config['name']}/", exist_ok=True)
    save_df.to_csv(f"{config['save_path']}/{problem_config['name']}/results.csv", index=False)

    # Important note. For some tensorflow reason observations need to be float even for classification problems.
    # Additional note. Tf variance can only accept real numbers
    init_dataset = Dataset(query_points=tf.cast(query_points, tf.float64),
                           observations=tf.cast(observations, tf.float64))

    if config['experiment']['noise_free']:
        config['kernel_variance'] = 1e-6
    else:
        config['kernel_variance'] = None

    model = build_model(init_dataset, search_space, config)
    # TODO extend to batch_sampling at a later stage
    if config['experiment']['batch_sampling']:
        if 'num_query_points' in config['experiment']:
            num_query_points = config['experiment']['num_query_points']
        else:
            # Not implemented
            num_query_points = 3
    else:
        num_query_points = 1

    if config['problem'] in ['multiobjective', 'vlmop2']:
        acq = ExpectedHypervolumeImprovement()
        print("Multiobjective model using HyperVolumes")
    elif config['experiment']['classification']:
        acq = BayesianActiveLearningByDisagreement()
        print("using Classification BALD")
    else:
        if config['experiment']['acquisition']:
            if config['experiment']['acquisition'] == 'negative_predictive_mean':
                acq = NegativePredictiveMean()
            elif config['experiment']['acquisition'] == 'predictive_variance':
                acq = PredictiveVariance()
            else:
                msg = f"Acquisition [{config['experiment']['acquisition']}] not identified, defaulting to Predictive Variance (Global Search)"
                warnings.warn(msg)
                acq = PredictiveVariance()
        else:
            msg = f"No Acquisition specified in the config.json/experiment. Defaulting to Predictive Variance (Global Search)"
            warnings.warn(msg)
            acq = PredictiveVariance()

        # Sample close to a given threshold
        # acq = ExpectedFeasibility(threshold=-0.5)
        # Equivalent to Maximizing a function
        # acq = NegativePredictiveMean()
        # Minimize global variance
        # acq = PredictiveVariance()

    rule = EfficientGlobalOptimization(builder=acq,
                                       num_query_points=num_query_points
                                       )
    bo = BayesianOptimizer(observer=problem_config['name'],
                           search_space=search_space,
                           scaler=scaler,
                           track_state=True,
                           track_path=f"{config['save_path']}/{problem_config['name']}/",
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

        if observations.ndim <= 1:
            # TODO Need to fix this to allow batch sampling,
            #  but also multivariate modeling.
            if config['problem'] == 'multiobjective':
                observations = observations.reshape(-1, 2)
            else:
                observations = observations.reshape(-1, 1)

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
    parser.add_argument('--initiator', choices=['Python', 'MATLAB'], required=False,
                        help="Specify the initiator process: 'Python' or 'MATLAB'")
    parser.add_argument('--target', choices=['Python', 'MATLAB'], required=False,
                        help="Specify the target process to control: 'Python' or 'MATLAB'")

    #Unknown args allow for external calls from Streamlit or other frontends
    args, unknown_args = parser.parse_known_args()

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

    try:
        main(**merged_args)
    except Exception as e:
        print(f"Error occurred: {e}")
        sys.exit(1)

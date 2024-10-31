import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time
import warnings
from threading import Thread

import numpy as np
import pandas as pd
import tensorflow as tf
import trieste
from trieste.acquisition.function import BayesianActiveLearningByDisagreement, \
    NegativePredictiveMean, PredictiveVariance, ExpectedHypervolumeImprovement
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.acquisition.rule import OBJECTIVE
from trieste.data import Dataset

from common.utils import load_json
# https://github.com/secondmind-labs/trieste/blob/c6a039aa9ecf413c7bcb400ff565cd283c5a16f5/trieste/acquisition/function/__init__.py
from online_learning import build_model
from online_neuro.bayessian_optimizer import BayesianOptimizer
from online_neuro.utils import customMinMaxScaler, SearchSpacePipeline, run_matlab, fetch_data
from common.plotting import update_plot
import requests
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

#
def start_connection(connection_config, problem_config, debug_matlab=False):
    """
    @param connection_config:
    @param initiator:
    @param target:
    @return:
    """
    # Create a TCP/IP socket
    server_socket = socket.socket(family=socket.AF_INET,
                                  type=socket.SOCK_STREAM)
    print(f"Establishing connection at port {connection_config['ip']} with port {connection_config['port']}")
    # Bind the socket to the address and port
    server_socket.bind((connection_config['ip'], connection_config['port']))

    # Listen for incoming connections
    server_socket.listen(1)
    print("Waiting for a connection...")
    # Start Matlab process via engine and threading

    if connection_config['initiator'] == "Python":
        if connection_config['target'] == "MATLAB":
            if not debug_matlab:
                # Flag to allow for manually launching Matlab and debugging its end
                # If threading, Python launches the Matlab main, else, main needs to be manually launched
                matlab_payload = dict()
                matlab_payload['connection_config'] = connection_config.copy()
                matlab_payload['problem_name'] = problem_config['experiment']['name']
                matlab_payload['problem_type'] = problem_config['experiment']['type']
                #TODO, improve this
                omit_keys = ['name', 'type']
                other_keys = [k for k in problem_config.keys() if k not in omit_keys]
                if len(other_keys) > 0:
                    matlab_payload['problem_config'] = {k: problem_config[k] for k in other_keys}
                else:
                    raise Exception("Problem configuration contains no features to optimize")

                print('Starting Matlab with Payload:')
                print(matlab_payload)
                matlab_payload['matlab_initiates'] = False
                # TODO, needs to be specified somewhere else
                matlab_payload['script_path'] = "simulators/matlab/"
                t = Thread(target=run_matlab, kwargs={"matlab_script_path": matlab_payload['script_path'],
                                                      "matlab_function_name": "main",
                                                      **matlab_payload
                                                      })
                t.start()
            else:
                print("You are in debug mode, start Matlab manually")
        elif connection_config['target'] == "Python":
            msg = f"No implementation available for target {connection_config['target']}"
            raise NotImplementedError(msg)
        else:
            msg = f"No implementation available for target {connection_config['initiator']}"
            raise NotImplementedError(msg)

    else:
        msg = f"No implementation available for initiator {connection_config['initiator']}. {connection_config['initiator']} has no optimization routines."
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


def define_search_space(problem_config, scale_inputs: bool = True, scaler: str= 'minmax'):
    """
    @param scale_inputs: bool, I don't see a scenario where this wouldn't be the case.
    @param default_config: Information concerning the problem. More specifically, the upper and lower boundary
            for normalization purposes.
    @param problem_config: Information concerning the search space as defined by the user.
                The gui boundaries should be within the problem_config range.
                If not the case, these are ignored.
    @param scaler : string to specify the scaling type(Probably Sklearn can be used here instead)
    @return: Tuple (Trieste Search Space, scaler)

    """
    # TODO extend to handle categorical and boolean features (i.e. they have to bypass the feature normalization
    filtered_feats = [k for k in problem_config.keys() if (('min_value' in problem_config[k]) and ('max_value' in problem_config[k]))]

    # TODO bypass fixed features (when min_value == max_value, the feature is fixed)

    lower_bound = []
    upper_bound = []
    feature_names = []

    feature_map = {}
    counter = 0
    for i, key in enumerate(filtered_feats):
        feature_names.append(key)

        feature_map[key] = np.arange(counter, counter + len(problem_config[key]['min_value']))
        counter += len(problem_config[key]['min_value'])

        for j, (min_v, max_v) in enumerate(zip(problem_config[key]['min_value'], problem_config[key]['max_value'])):
            lower_bound.append(min_v)
            upper_bound.append(max_v)

    if scale_inputs:
        num_features = len(lower_bound)

        # TODO, implement other types of normalization
        if scaler == 'minmax':

            output_range = (-1, 1)

            lb = num_features * [output_range[0]]
            ub = num_features * [output_range[1]]

            search_space = trieste.space.Box(lower=lb,
                                             upper=ub)

            scaler = customMinMaxScaler(feature_min=lower_bound,
                                        feature_max=upper_bound,
                                        output_range=output_range)
        else:
            raise NotImplementedError(f"{scaler} has not been implemented")
    else:

        search_space = trieste.space.Box(lower=lower_bound,
                                         upper=upper_bound)
        scaler = None

    pipeline = SearchSpacePipeline(search_space=search_space,
                                   mapping=feature_map,
                                   feature_names=feature_names,
                                   scaler=scaler)

    return pipeline


def main(connection_config: dict, model_config: dict, path_config: dict,
         problem_config: dict, *args, **kwargs) -> None:
    """
    @param connection_config:
    @param model_config:
    @param path_config:
    @param problem_config:
    @param args:
    @param kwargs:
    @return:
    """

    default_dict = load_json("config.json")

    print("problem_config")
    print(problem_config)

    connection_config = {**default_dict['connection_config'], **connection_config}
    model_config = {**default_dict['model_config'], **model_config}
    path_config = {**default_dict['path_config'], **path_config}
    #problem_config = {**default_dict['problem_config'], **problem_config}

    #Port Flask indicates that a UI is listening to the process. We can send partial results and other info to this interface
    port_flask = connection_config.get('port_flask', False)
    if port_flask:
        post_url = f"http://localhost:{port_flask}"
    # TODO
    #  - verify here that the problem can be solved by the simulator (target)
    # i.e. matlab, axonsim, neuron, python, each need to have a simulator file for the given problem.
    #  - verify that the model type can solve the problem type
    # i.e. prevent multiobjective/regression to be solved with classification, etc...

    server_socket, client_socket = start_connection(connection_config=connection_config,
                                                    problem_config=problem_config,
                                                    debug_matlab=False)

    handshake_check(client_socket=client_socket)

    ## First message
    # Receives feature names, target_names(?), Upper and lower_boundaries.
    # This should probably also pass constraints, although this needs to be thought still.
    # TODO this line won't be needed anymore, as search space is now defined from the UI and not the problem config
    received_data = client_socket.recv(1024).decode()
    problem_specs = json.loads(received_data)


    print("Loaded problem config")
    print(problem_config)

    search_space_pipe = define_search_space(problem_config=problem_config,
                                            scale_inputs=model_config["scale_inputs"])

    ## First batch requested to the simulator.
    # Initialize GP and search strategy with first batch of data.

    # TODO Optional. Add other sampling methods such as LHS (Halton is in Trieste but not needed).
    qp_dict, qp_array = search_space_pipe.sample(model_config['init_samples'],
                                                 sample_method='sobol')

    def list_dict_process(some_list):
        new_list = []
        for element in some_list:
            new_list.append({k:v.tolist() for k,v in element.items()})
        return new_list

    qp_json = list_dict_process(qp_dict)

    response = {'message': 'first queried points using Sobol method',
                'query_points': qp_json}

    response_json = json.dumps(response) + "\n"
    client_socket.sendall(response_json.encode())
    received_data = fetch_data(client_socket)

    print("First data package:", received_data)
    received_data = pd.DataFrame(received_data)
    observations = received_data['init_response'].values

    if isinstance(observations[0], list):
        observations = np.vstack(observations)

    observations = np.atleast_2d(observations).T

    init_data_size = len(qp_array)
    column_names = search_space_pipe.feature_names_ixs()
    results_df = pd.DataFrame(qp_array, columns=column_names)

    if observations.shape[1] > 1:
        for i in range(observations.shape[1]):
            results_df[f'response_{i}'] = observations[:, i]
    else:
        results_df['response'] = observations

    if port_flask:
        json_data = results_df.to_json(orient='records')
        response = requests.post(post_url+"/update_data", json=json_data)
        # Check the response
        if response.status_code == 200:
            print(response.json())
        else:
            print("Error updating data:", response.status_code, response.text)

    # TODO Complete file naming
    os.makedirs(f"{path_config['save_path']}/{problem_config['experiment']['name']}/", exist_ok=True)
    results_df.to_csv(f"{path_config['save_path']}/{problem_config['experiment']['name']}/results.csv", index=False)

    # Important note. For some tensorflow reason observations need to be float even for classification problems.
    # Additional note. Tf variance can only accept real numbers

    init_dataset = Dataset(query_points=tf.cast(qp_array, tf.float64),
                           observations=tf.cast(observations, tf.float64))

    if model_config['noise_free']:
        model_config['kernel_variance'] = 1e-6
    else:
        model_config['kernel_variance'] = None

    model = build_model(init_dataset,
                        search_space_pipe.search_space,
                        model_config)

    # TODO extend to batch_sampling at a later stage
    if model_config['batch_sampling']:
        if 'num_query_points' in model_config:
            if model_config['num_query_points'] == 1:
                warnings.warn(f"Batch querying was specified, but config specifies only 1 sample")
        else:
            warnings.warn(
                f"Batch querying was specified, but config filed does not specify batch size, defaulting to three")
            model_config['num_query_points'] = 3

    #Todo
    if problem_config['experiment']['type'] in ['multiobjective', 'moo']:
        acq = ExpectedHypervolumeImprovement()
        print("Multiobjective model using HyperVolumes")
    elif problem_config['experiment']['type'] in ['classification']:
        acq = BayesianActiveLearningByDisagreement()
        print("using Classification BALD")
    elif problem_config['experiment']['type'] in ['regression']:
        if 'acquisition' in problem_config:
            if problem_config['acquisition'] == 'negative_predictive_mean':
                acq = NegativePredictiveMean()
            elif problem_config['acquisition'] == 'predictive_variance':
                acq = PredictiveVariance()
            else:
                msg = f"Acquisition [{problem_config['experiment']['acquisition']}] not identified, defaulting to Predictive Variance (Global Search)"
                warnings.warn(msg)
                acq = PredictiveVariance()
        else:
            msg = f"No Acquisition specified in the problem configuration / --flags. Default is minimizing Predictive Variance"
            warnings.warn(msg)
            acq = PredictiveVariance()
    else:
        msg = f"No Acquisition specified in the problem configuration / --flags. Default is minimizing Predictive Variance"
        warnings.warn(msg)
        acq = PredictiveVariance()

        # TODO, extend acquisitions as required
        # Sample close to a given threshold
        # acq = ExpectedFeasibility(threshold=-0.5)
        # Equivalent to Maximizing a function
        # acq = NegativePredictiveMean()
        # Minimize global variance
        # acq = PredictiveVariance()

    rule = EfficientGlobalOptimization(builder=acq,
                                       num_query_points=model_config['num_query_points']
                                       )
    bo = BayesianOptimizer(observer=problem_config['experiment']['name'],
                           search_space=search_space_pipe.search_space,
                           scaler=search_space_pipe.scaler,
                           track_state=True,
                           track_path=f"{path_config['save_path']}/{problem_config['experiment']['name']}/",
                           acquisition_rule=rule)

    qp = bo.request_query_points(datasets=init_dataset,
                                 models=model,
                                 fit_initial_model=True,
                                 fit_model=True)

    qp_dict, qp_array = search_space_pipe.inverse_transform(qp, with_array=True)
    print("First qp")

    print(qp_dict)

    terminate_flag = False
    if qp_dict is None:
        raise Exception("Terminated before optimization started \n no query points were retrieved")

    with_plots = True
    count = 0

    # TODO send stop signals from both ends upon certain conditions (i.e. no improvement, reached goal,
    # buget limit, etc...)

    # TODO, for toy-problems, add the option of generating performance results (i.e. Accuracy/RMSE over
    # updates

    # TODO, for real-world problems, generate the GridSearch solution, store (within GIT?) And use to
    # generate example figures of online learning efficiency

    # TODO, plotting functions for Pareto

    # TODO, plotting functions for 3D search shapes (see Notebook in NeuroAAA repo)

    while not terminate_flag:
        # Counter is used to save figures with different names. If kept constant it overwrites the figure.
        count += 1

        # Check if termination signal received from MATLAB
        # Respond back
        qp_json = list_dict_process(qp_dict)
        message = {"query_points": qp_json,
                   "terminate_flag": terminate_flag}

        print(message)
        response_json = json.dumps(message) + "\n"
        client_socket.sendall(response_json.encode())

        if "terminate_flag" in received_data:
            terminate_flag = received_data['terminate_flag']
            print("Termination signal received from MATLAB \n Saving results... \n Closing connection...")

        received_data = client_socket.recv(1024).decode()
        received_data = json.loads(received_data)

        # TODO, this routine may be combined or redundant if plot-flask?
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

        observations = np.array([received_data['observations']])
        observations = np.atleast_2d(observations).T

        # Save results
        new_sample = pd.DataFrame(qp_array, columns=column_names)

        if observations.shape[1] > 1:
            for i in range(observations.shape[1]):
                new_sample[f'response_{i}'] = observations[:, i]
        else:
            new_sample['response'] = observations

        results_df = pd.concat([results_df, new_sample], ignore_index=True)
        # TODO, a line by line write to csv is more efficient than rewriting the whole file
        results_df.to_csv(f"{path_config['save_path']}/{problem_config['experiment']['name']}/results.csv", index=False)

        if port_flask:
            json_data = new_sample.to_json(orient='records')
            response = requests.post(post_url + "/update_data", json=json_data)
            # Check the response
            if response.status_code == 200:
                print(response.json())
            else:
                print("Error updating data:", response.status_code, response.text)

        print(qp_array.shape, observations.shape)
        print(qp_array)

        print(observations)
        bo.optimize_step(query_points=qp_array,
                         observer_output=observations)

        final_data = bo.result.try_get_final_datasets()[OBJECTIVE]
        final_models = bo.result.try_get_final_models()[OBJECTIVE]

        qp = bo.request_query_points(datasets=final_data,
                                     models=final_models,
                                     fit_initial_model=False,
                                     fit_model=True)

        qp_dict, qp_array = search_space_pipe.inverse_transform(qp, with_array=True)

    # Close the connection
    client_socket.close()
    server_socket.close()

def convert_to_numeric(value):
    """Convert a comma-separated string to a list of floats or a single float."""
    if isinstance(value, list):
        return value

    if ',' in value:
        return [float(v) for v in value.split(',')]
    elif value == 'True':
        return True
    elif value == 'False':
        return False
    else:
        return float(value)


def process_dict(temp_dict):
    for k, v in temp_dict.items():
        if v is None:
            temp_dict[k] = {}
        elif isinstance(v, str):
            # First, try to parse the string as JSON
            try:
                temp_dict[k] = json.loads(v)
            except json.JSONDecodeError:
                # If it's not valid JSON, check if it's a valid file path
                if len(v) > 255:  # Length check for unusually long strings
                    return 'invalid file'
                elif Path(v).exists():  # If it's a valid file path
                    temp_dict[k] = load_json(v)
                else:
                    print(type(v))
                    print(v)
                    raise Exception("This doesn't seem to be a path nor a valid JSON")
        else:
            print(f"Unhandled type for value {v}")

    return temp_dict


def process_args(args, unknown_args):
    """
    @param args:
    @param unknown_args:
    @return:
    """
    if args.config:
        if isinstance(args.config, str):
            config = load_json(args.config)
        elif isinstance(args.config, dict):
            config = args.config
        else:
            raise NotImplementedError

        connection_config = config.get('connection_config', {})
        model_config = config.get('model_config', {})
        path_config = config.get('path_config', {})
        problem_config = config.get('problem_config', {})
        other_params = dict()

    elif args.connection_config or args.model_config or \
            args.problem_config or args.path_config:

        temp_dict = dict()
        if isinstance(args, dict):
            for k in ['connection_config', 'model_config', 'path_config','problem_config']:
                temp_dict[k] = args.get(k, {})
        else:
            for k in ['connection_config', 'model_config', 'path_config', 'problem_config']:
                temp_dict[k] = getattr(args, k, {})

        temp_dict = process_dict(temp_dict=temp_dict)

        connection_config = temp_dict['connection_config']
        model_config = temp_dict['model_config']
        path_config = temp_dict['path_config']
        problem_config = temp_dict['problem_config']

        other_params = dict()

    else:
        args_dict = vars(args)
        print(f"Known args: {args_dict}")
        print(f"Unknown args: {unknown_args}")

        unknown_args_dict = {}
        i = 0
        while i < len(unknown_args):
            if unknown_args[i].startswith('--'):
                key = unknown_args[i][2:]  # Remove '--' from flag
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

        connection_config = dict()
        problem_config = dict()
        model_config = dict()
        path_config = dict()
        other_params = dict()

        for k in ['initiator', 'target']:
            if k in args_dict:
                connection_config[k] = args_dict[k]

        for k, v in unknown_args_dict.items():
            try:
                key_split = k.split("_")
                k_beginning = key_split[0]
                actual_key = '_'.join(key_split[1:])
            except Exception as e:
                warnings.warn(f"{k} seems to not be a valid formatted flag, passing it directly to main")

            if k_beginning == 'conn':
                connection_config[actual_key] = v
            elif k_beginning == 'prob':
                problem_config[actual_key] = convert_to_numeric(v)
            elif k_beginning == 'model':
                model_config[actual_key] = convert_to_numeric(v)
            elif k_beginning == 'path':
                path_config[actual_key] = v
            else:
                other_params[k] = v

    def convert_string_numeric(data):
        if isinstance(data, dict):
            return {key:convert_string_numeric()}

    return connection_config, model_config, path_config, problem_config, other_params


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Controller to start Python or MATLAB process.")

    group1 = parser.add_argument_group('Global configuration')
    group1.add_argument('--config', type=str, help="Path to or JSON of the global configuration file")

    group2 = parser.add_argument_group('Specific configurations')

    group2.add_argument('--connection_config', type=str, help="Path to or json of the connection's configuration file")
    group2.add_argument('--model_config', type=str, help="Path to or json of the model's configuration file")
    group2.add_argument('--problem_config', type=str, help="Path to or json of the problems's configuration file")
    group2.add_argument('--path_config', type=str, help="Path to or json of paths's configuration file")

    parser.add_argument('--initiator', choices=['Python'], required=False,
                        help="Specify the initiator process: 'Python'. 'Matlab has been deprecated'")
    parser.add_argument('--target', choices=['Python', 'MATLAB'], required=False,
                        help="Specify the target process to control: 'Python' or 'MATLAB'")
    parser.add_argument('--flask_port', required=False,
                        help="If provided, partial results are sent to the flask server for plotting and reporting")

    # Unknown args allow for external calls from Streamlit or other frontends
    args, unknown_args = parser.parse_known_args()

    group2_args = [args.connection_config, args.model_config, args.problem_config, args.path_config]
    # Check that either --config is provided or any of group2, but not both
    if args.config and any(group2_args):
        parser.error("--config cannot be used with any of --connection_config, --model_config, --problem_config, or --path_config")
    elif not args.config and not any(group2_args):
        parser.error("You must provide either --config or at least one of --connection_config, --model_config, --problem_config, or --path_config")

    try:
        conn_config, mod_config, path_config, prob_config, other_params = process_args(args, unknown_args)
    except Exception as e:
        print(e)

    print("Rearranged configs:")
    print(conn_config, mod_config, path_config, prob_config, other_params)

    try:
        main(connection_config=conn_config,
             model_config=mod_config,
             path_config=path_config,
             problem_config=prob_config,
             **other_params)

    except Exception as e:
        print(f"Error occurred: {e}")
        sys.exit(1)

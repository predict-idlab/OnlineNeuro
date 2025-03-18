# online_neuro/server_side.py
import argparse
import atexit
import json
import sys
import time
import timeit
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from gpflow.likelihoods import Bernoulli as bern
from trieste.acquisition.function import BayesianActiveLearningByDisagreement, \
    NegativePredictiveMean, PredictiveVariance, ExpectedHypervolumeImprovement
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.acquisition.rule import OBJECTIVE
from trieste.data import Dataset

from common.plotting import update_plot
from common.utils import load_json
from online_learning import build_model
from online_neuro.bayessian_optimizer import AskTellOptimizerHistory
from online_neuro.connection_utils import start_connection, handshake_check, cleanup_server_socket
from online_neuro.utils import define_scaler_search_space
from online_neuro.utils import fetch_data, array_to_list_of_dicts

GRID_POINTS = 10
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


def plot_flask(sample: pd.DataFrame, plot_type: str, post_url: str)->None:
    """
    Send data to plot to Flask server.
    @param sample: pd.DataFrame with features and observations
    @param plot_type: string specifying the plot type to serve
    @param post_url: the url specifying which plot to update
    @return:
    """
    json_message = dict()
    json_message['data'] = sample.to_json(orient='records')
    json_message['plot_type'] = plot_type
    response = requests.post(post_url + '/update_data', json=json_message)
    # Check the response
    if response.status_code == 200:
        print(response.json())
    else:
        print('Error updating data:', response.status_code, response.text)


def main(connection_config: dict, model_config: dict, path_config: dict,
         problem_config: dict, **kwargs) -> None:
    """
    @param connection_config:
    @param model_config:
    @param path_config:
    @param problem_config:
    @param kwargs:
    @return:
    """

    default_dict = load_json('config.json')

    connection_config = {**default_dict['connection_config'], **connection_config}
    model_config = {**default_dict['model_config'], **model_config}
    path_config = {**default_dict['path_config'], **path_config}

    # Port Flask indicates that a UI is listening to the process.
    # We can send partial results and other info to this interface
    port_flask = connection_config.get('port_flask', False)

    if port_flask:
        post_url = f'http://localhost:{port_flask}'
    # TODO
    #  - verify here that the problem can be solved by the simulator (target)
    # i.e. matlab, axonsim, neuron, python, each need to have a simulator file for the given problem.
    #  - verify that the model type can solve the problem type
    # i.e. prevent multiobjective or regression to be solved with classification, etc...
    server_socket, client_socket = start_connection(connection_config=connection_config,
                                                    problem_config=problem_config)

    handshake_check(client_socket=client_socket, verbose=True)

    atexit.register(cleanup_server_socket, client_sock=client_socket, server_sock=server_socket)
    package_size = connection_config['SizeLimit']

    print("problem config")
    print(problem_config)

    search_space, scaler, feat_dict = define_scaler_search_space(problem_config=problem_config,
                                                                 scale_inputs=model_config['scale_inputs'])

    feature_names = list(feat_dict['variable_features'].keys())
    fixed_features = feat_dict['fixed_features']

    print('Fixed features')
    print(fixed_features)
    print('Variable features')
    print(feature_names)
    msg = json.dumps({'Fixed_features': fixed_features}) + '\n'
    client_socket.sendall(msg.encode())

    if len(feature_names) == 2:
        x0 = np.linspace(scaler.feature_min[0], scaler.feature_max[0], GRID_POINTS)
        x1 = np.linspace(scaler.feature_min[1], scaler.feature_max[1], GRID_POINTS)
        meshgrid_x, meshgrid_y = np.meshgrid(x0, x1)
        plot_inputs = np.column_stack([meshgrid_x.ravel(), meshgrid_y.ravel()])

        x0_scaled = np.linspace(scaler.output_min[0], scaler.output_max[1], GRID_POINTS)
        x1_scaled = np.linspace(scaler.output_min[0], scaler.output_max[1], GRID_POINTS)
        meshgrid_x, meshgrid_y = np.meshgrid(x0_scaled, x1_scaled)
        model_inputs = np.column_stack([meshgrid_x.ravel(), meshgrid_y.ravel()])

    # TODO Optional. Add other sampling methods such as LHS (Halton is in Trieste but not needed).
    qp_orig = search_space.sample_method(model_config['init_samples'], sampling_method='sobol')
    qp_orig = qp_orig.numpy()

    if scaler:
        qp = scaler.inverse_transform(qp_orig)
    else:
        qp = qp_orig

    qp_json = array_to_list_of_dicts(qp, feature_names)

    print('First batch')
    print(qp_json)
    response = {'message': 'first queried points using Sobol method',
                'query_points': qp_json}

    response_json = json.dumps(response) + '\n'
    client_socket.sendall(response_json.encode())

    received_data = fetch_data(client_socket)
    print(received_data)

    observations = [r['observations'] for r in received_data]

    # TODO check if this is correct for MOO
    if isinstance(observations[0], list):
        observations = np.vstack(observations)

    observations = np.atleast_2d(observations).T

    init_data_size = len(qp)
    results_df = pd.DataFrame(qp, columns=feature_names)

    if observations.shape[1] > 1:
        response_cols = []
        for i in range(observations.shape[1]):
            results_df[f'response_{i}'] = observations[:, i]
            response_cols.append(f'response_{i}')
    else:
        results_df['response'] = observations
        response_cols = ['response']

    save_path = Path(f"{path_config['save_path']}") / f"{problem_config['experiment']['name']}"
    csv_path = Path(save_path) / 'results.csv'
    model_store_path = Path(save_path) / 'models'

    save_path.mkdir(parents=True, exist_ok=True)
    model_store_path.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(csv_path, index=False)

    # @Note. For some tensorflow reason observations need to be float even for classification problems.
    init_dataset = Dataset(query_points=tf.cast(qp_orig, tf.float64),
                           observations=tf.cast(observations, tf.float64))

    if model_config['noise_free']:
        model_config['kernel_variance'] = 1e-6
    else:
        model_config['kernel_variance'] = None

    model = build_model(init_dataset,
                        search_space,
                        model_config)

    # TODO extend to batch_sampling at a later stage
    if model_config['batch_sampling']:
        if 'num_query_points' in model_config:
            if model_config['num_query_points'] == 1:
                msg = f"""'Batch querying was specified to {model_config['num_query_points']},
                but config specifies only 1 sample'"""
                warnings.warn(msg)
        else:
            msg = 'Batch querying was specified, but not batch size was specified, defaulting to three'
            warnings.warn(msg)
            model_config['num_query_points'] = 3

    if problem_config['experiment']['type'] in ['multiobjective', 'moo']:
        acq = ExpectedHypervolumeImprovement()
        print('Multi-objective model using HyperVolumes')
    elif problem_config['experiment']['type'] in ['classification']:
        acq = BayesianActiveLearningByDisagreement()
        print('using Classification BALD')
    elif problem_config['experiment']['type'] in ['regression']:
        if 'acquisition' in problem_config:
            if problem_config['acquisition'] == 'negative_predictive_mean':
                acq = NegativePredictiveMean()
            elif problem_config['acquisition'] == 'predictive_variance':
                acq = PredictiveVariance()
            else:
                msg = f"""Acquisition {problem_config['experiment']['acquisition']}] not identified,
                defaulting to Predictive Variance (Global Search)"""
                warnings.warn(msg)
                acq = PredictiveVariance()
        else:
            msg = 'No Acquisition specified. Defaulting to minimizing Predictive Variance'
            warnings.warn(msg)
            acq = PredictiveVariance()
    else:
        msg = 'No Acquisition specified. Defaulting to minimizing Predictive Variance'
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
    bo = AskTellOptimizerHistory(observer=problem_config['experiment']['name'],
                                 datasets=init_dataset,
                                 search_space=search_space,
                                 models=model,
                                 acquisition_rule=rule,
                                 fit_model=True,
                                 overwrite=True,
                                 track_path=model_store_path
                                 )

    if port_flask:
        plot_flask(results_df, 'pairplot', post_url)

        if 'time' in received_data[0]:
            # Only sending the last case for plotting
            pulse_data = pd.DataFrame()
            pulse_data['pulse_a'] = received_data[-1]['pulse_a']
            pulse_data['pulse_b'] = received_data[-1]['pulse_b']
            pulse_data['time'] = received_data[-1]['time']

            plot_flask(pulse_data, 'pulses', post_url)

        if len(feature_names) == 2:
            # TODO handle this. Only if the problem is a classification problem!
            # TODO this plot will be used seldomly, so it may make sense to load in front-end only if it will be used.

            mean, variance = bo.models[OBJECTIVE].predict(model_inputs)  # Predict mean and variance

            mean = bern().invlink(mean)

            Z = mean.numpy().reshape(meshgrid_x.shape)
            Z_var = variance.numpy().reshape(meshgrid_x.shape)
            json_message = dict()
            data = pd.DataFrame(np.column_stack([x0.squeeze(), x1.squeeze()]), columns=['x', 'y'])

            json_message['z'] = Z.tolist()
            json_message['z_var'] = Z_var.tolist()
            json_message['data'] = data.to_json(orient='records')
            json_message['plot_type'] = 'contour'

            response = requests.post(post_url + '/update_data',
                                     json=json_message,
                                     headers={'Content-Type': 'application/json'}
                                     )
            # Check the response
            if response.status_code != 200:
                warnings.warn('Error updating data:', response.status_code, response.text)

    print('Init dataset')
    print(init_dataset)
    terminate_flag = False

    # TODO send stop signals from both ends upon certain conditions (i.e. no improvement, reached goal,
    # buget limit, etc...)
    # TODO, for toy-problems, add the option of generating performance results (i.e. Accuracy/RMSE over
    # updates
    # TODO, for real-world problems, generate the GridSearch solution, store (within GIT?) And use to
    # generate example figures of online learning efficiency
    # TODO, plotting functions for Pareto
    # TODO, plotting functions for 3D search shapes (see Notebook in NeuroAAA repo)

    with_plots = False
    count = 0
    exit_message = None

    # TODO, use this
    start = timeit.default_timer()

    while not terminate_flag:
        # Counter is used to save figures with different names. If kept constant it overwrites the figure.
        count += 1

        qp_orig = bo.ask_and_save()
        if qp is None:
            raise Exception(f'Terminated at step {count} before optimization started')

        qp_orig = qp_orig.numpy()
        if scaler:
            qp = scaler.inverse_transform(qp_orig)
        else:
            qp = qp_orig

        # Respond back
        qp_json = array_to_list_of_dicts(qp, feature_names)
        message = {'query_points': qp_json,
                   'terminate_flag': terminate_flag}

        print(f'Count {count}')
        response_json = json.dumps(message) + '\n'
        client_socket.sendall(response_json.encode())

        received_data = fetch_data(client_socket)
        print('Received')
        print(received_data)

        # Check if termination signal received from simulator
        if 'terminate_flag' in received_data[0]:
            terminate_flag = received_data['terminate_flag']
            exit_message = 'Termination signal received from MATLAB \n Saving results... \n Closing connection...'

        observations = [r['observations'] for r in received_data]
        observations = np.atleast_2d(observations).T

        # Save results
        new_sample = pd.DataFrame(qp, columns=feature_names)

        if observations.shape[1] > 1:
            for i in range(observations.shape[1]):
                new_sample[response_cols[i]] = observations[:, i]
        else:
            new_sample[response_cols] = observations

        results_df = pd.concat([results_df, new_sample], ignore_index=True)

        # TODO, a line by line write to csv is more efficient than rewriting the whole file
        results_df.to_csv(csv_path, index=False)

        # TODO, this routine may be combined or redundant if plot-flask?
        if with_plots:
            update_plot(bo,
                        search_space=search_space,
                        scaler=scaler,
                        initial_data=(results_df.iloc[:init_data_size][feature_names],
                                      results_df.iloc[:init_data_size][response_cols]),
                        sampled_data=(results_df.iloc[init_data_size:][feature_names],
                                      results_df.iloc[init_data_size:][response_cols]),
                        plot_ground_truth=None, ground_truth_function=None,
                        count=count)
            time.sleep(0.1)

        if port_flask:
            plot_flask(new_sample, 'pairplot', post_url)
            if 'time' in received_data[0]:
                pulse_data = pd.DataFrame()
                pulse_data['pulse_a'] = received_data[-1]['pulse_a']
                pulse_data['pulse_b'] = received_data[-1]['pulse_b']
                pulse_data['time'] = received_data[-1]['time']

                plot_flask(pulse_data, 'pulses', post_url)
            if len(feature_names) == 2:
                mean, variance = bo.models[OBJECTIVE].predict(model_inputs)  # Predict mean and variance
                # only if the problem is a classification problem!
                mean = bern().invlink(mean)

                Z = mean.numpy().reshape(meshgrid_x.shape)
                Z_var = variance.numpy().reshape(meshgrid_x.shape)
                json_message = dict()
                # Only the first time, to see axis in their original scale (Maybe this should be ticks, to prevent
                # plot distortions?
                data = pd.DataFrame(plot_inputs, columns=['x', 'y'])
                json_message['z'] = Z.tolist()
                json_message['z_var'] = Z_var.tolist()
                json_message['data'] = data.to_json(orient='records')
                json_message['plot_type'] = 'contour'

                response = requests.post(post_url + '/update_data',
                                         json=json_message,
                                         headers={'Content-Type': 'application/json'}
                                         )
                # Check the response
                if response.status_code == 200:
                    print(response.json())
                else:
                    print('Error updating data:', response.status_code, response.text)

        tagged_output = Dataset(query_points=tf.cast(qp_orig, tf.float64),
                                observations=tf.cast(observations, tf.float64))

        bo.tell(tagged_output)

    print(exit_message)
    cleanup_server_socket(client_sock=client_socket, server_sock=server_socket)


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
            print(f'Unhandled type for value {v}')

    return temp_dict


def process_args(args, unknown_args):
    """
    TODO update this function
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
            for k in ['connection_config', 'model_config', 'path_config', 'problem_config']:
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
        print(f'Known args: {args_dict}')
        print(f'Unknown args: {unknown_args}')

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

        for k in ['target']:
            if k in args_dict:
                connection_config[k] = args_dict[k]

        for k, v in unknown_args_dict.items():
            try:
                key_split = k.split('_')
                k_beginning = key_split[0]
                actual_key = '_'.join(key_split[1:])
            except ValueError as exception:
                msg = exception(f'{k}. seems to not be a valid formatted flag, passing it directly to main')
                warnings.warn(msg)

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

    return connection_config, model_config, path_config, problem_config, other_params


if __name__ == '__main__':
    print('PYTHONPATH:', sys.path)
    parser = argparse.ArgumentParser(description='Controller to start Python or MATLAB process.')

    group1 = parser.add_argument_group('Global configuration')
    group1.add_argument('--config', type=str, help='Path to or JSON of the global configuration file')

    group2 = parser.add_argument_group('Specific configurations')

    group2.add_argument('--connection_config', type=str, help="Path to or json of the connection's configuration file")
    group2.add_argument('--model_config', type=str, help="Path to or json of the model's configuration file")
    group2.add_argument('--problem_config', type=str, help="Path to or json of the problems's configuration file")
    group2.add_argument('--path_config', type=str, help="Path to or json of paths's configuration file")

    parser.add_argument('--target', choices=['Python', 'MATLAB'], required=False,
                        help="Specify the target simulator: 'Python' or 'MATLAB'")
    parser.add_argument('--flask_port', required=False,
                        help='If provided, partial results are sent to the flask server for plotting and reporting')

    # Unknown args allow for external calls from Streamlit or other frontends
    args, unknown_args = parser.parse_known_args()

    group2_args = [args.connection_config, args.model_config, args.problem_config, args.path_config]
    # Check that either --config is provided or any of group2, but not both
    if args.config and any(group2_args):
        parser.error('--config cannot be used with any of --connection_config, --model_config, --problem_config, or --path_config')
    elif not args.config and not any(group2_args):
        parser.error('You must provide either --config or at least one of --connection_config, --model_config, --problem_config, or --path_config')

    try:
        conn_config, mod_config, path_config, prob_config, other_params = process_args(args, unknown_args)
    except Exception as e:
        print(e)

    print('Rearranged configs:')
    print(conn_config, mod_config, path_config, prob_config, other_params)

    try:
        main(connection_config=conn_config,
             model_config=mod_config,
             path_config=path_config,
             problem_config=prob_config,
             **other_params)

    except Exception as e:
        print(f'Error occurred: {e}')
        sys.exit(1)

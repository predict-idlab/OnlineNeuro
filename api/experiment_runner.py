# api/experiment_runner.py
# Main function that carries the iterative optimization loop.
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import tensorflow as tf

# Online Neurop imports
from frontend.utils.plot_helpers import plot_flask

# Trieste imports
from trieste.acquisition.function.active_learning import (
    BayesianActiveLearningByDisagreement,
    PredictiveVariance,
)
from trieste.acquisition.function.function import NegativePredictiveMean
from trieste.acquisition.function.multi_objective import ExpectedHypervolumeImprovement
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.data import Dataset
from trieste.observer import OBJECTIVE

# UI and Config imports
from api.config_utils import prepare_experiment_paths, setup_configs
from networking.connection_utils import cleanup_server_socket, setup_networking
from online_neuro.bayessian_optimizer import AskTellOptimizerHistory
from online_neuro.constants import ProblemType
from online_neuro.online_learning import build_model
from online_neuro.utils import (
    array_to_list_of_dicts,
    define_scaler_search_space,
    fetch_data,
)

# Global variable for plotting resolution.
GRID_POINTS = 10

# TODO, extend acquisitions as required. Currently Trieste includes the following:
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

# TODO send stop signals from both ends upon certain conditions (i.e. no improvement, reached goal,
# buget limit, etc...)
# TODO, for toy-problems, add the option of generating performance results (i.e. Accuracy/RMSE over
# updates
# TODO, for real-world problems, generate the GridSearch solution, store (within GIT?) And use to
# generate example figures of online learning efficiency
# TODO, plotting functions for Pareto
# TODO, plotting functions for 3D search shapes (see Notebook in NeuroAAA repo)


def get_post_url(connection_config: dict) -> str:
    """Constructs the Flask post URL if a port is specified."""
    port_flask = connection_config.get("port_flask")
    return f"http://localhost:{port_flask}" if port_flask else ""


def get_problem_class(exp_type: str) -> ProblemType:
    """Determines the problem type from the configuration."""
    exp_type = exp_type.lower()
    if exp_type in ["multiobjective", "moo"]:
        return ProblemType.MULTIOBJECTIVE
    if exp_type in ["classification"]:
        return ProblemType.CLASSIFICATION
    return ProblemType.REGRESSION


def select_acquisition_function(problem_type: str, acq_name: str):
    """Selects the appropriate acquisition function based on the problem type."""
    problem_class = get_problem_class(problem_type)

    acq_map = {
        ProblemType.MULTIOBJECTIVE: ExpectedHypervolumeImprovement,
        ProblemType.CLASSIFICATION: BayesianActiveLearningByDisagreement,
    }

    if problem_class in acq_map:
        print(f"Using acquisition function for {problem_class.name}")
        return acq_map[problem_class]()

    # Handle Regression case with more options
    if problem_class == ProblemType.REGRESSION:
        if acq_name == "negative_predictive_mean":
            print("Using Negative Predictive Mean for regression.")
            return NegativePredictiveMean()

        if acq_name != "predictive_variance":
            warnings.warn(
                f"Acquisition '{acq_name}' not identified for regression. Defaulting to Predictive Variance."
            )

        print("Using Predictive Variance for regression.")
        return PredictiveVariance()

    # Fallback, though the logic above should cover all cases
    warnings.warn(
        "Could not determine a specific acquisition function. Defaulting to Predictive Variance."
    )
    return PredictiveVariance()


def update_flask_ui(
    post_url: str,
    new_data_df: pd.DataFrame,
    received_data: list,
    bo_model=None,
    plot_inputs: dict | None = None,
):
    """Sends various data updates to the Flask UI.
    Note: alternatively or additionally, we can use  update_plot to store the results locally.

    """
    assert plot_inputs is None or "model_inputs" in plot_inputs
    if not post_url:
        return

    plot_flask(new_data_df, "pairplot", post_url)

    if received_data and "time" in received_data[0]:
        last_pulse = received_data[-1]
        pulse_data = pd.DataFrame(
            {
                "pulse_a": last_pulse["pulse_a"],
                "pulse_b": last_pulse["pulse_b"],
                "time": last_pulse["time"],
            }
        )
        plot_flask(pulse_data, "pulses", post_url)

    if (
        bo_model
        and plot_inputs is not None
        and plot_inputs["model_inputs"].shape[1] == 2
    ):
        # Assuming model_inputs is calculated from plot_inputs based on scaling
        # Changed this from predict to predict_y, so no invlink is needed now
        mean, variance = bo_model.predict_y(plot_inputs["model_inputs"])

        json_message = {
            "z": mean.numpy().reshape(plot_inputs["shape"]).tolist(),
            "z_var": variance.numpy().reshape(plot_inputs["shape"]).tolist(),
            "data": plot_inputs["original_grid"].to_json(orient="records"),
            "plot_type": "contour",
        }
        try:
            response = requests.post(f"{post_url}/update_data", json=json_message)
            response.raise_for_status()  # Raises an exception for bad status codes
        except requests.exceptions.RequestException as e:
            warnings.warn(f"Error updating Flask UI: {e}")


def run_optimization_loop(
    bo,
    client_socket,
    scaler,
    feature_names,
    response_cols,
    results_df,
    csv_path,
    post_url,
    plot_inputs_2d,
    verbose=False,
):
    """The main ask-tell optimization loop."""
    count = 0
    while True:
        print(f"count: {count+1}")
        count += 1
        if verbose:
            print(f"Optimization step {count}")

        # ASK
        qp_orig = bo.ask_and_save()
        if qp_orig is None:
            print(f"Optimizer returned None at step {count}. Terminating.")
            break
        qp_orig = qp_orig.numpy()

        # Inverse transform for the simulator
        qp = scaler.inverse_transform(qp_orig) if scaler else qp_orig

        # SEND query to simulator
        qp_json = array_to_list_of_dicts(qp, feature_names)
        message = {"query_points": qp_json, "terminate_flag": False}
        client_socket.sendall((json.dumps(message) + "\n").encode())

        # RECEIVE observations
        received_data = fetch_data(client_socket)
        if verbose:
            print("Received:", received_data)

        if not received_data or received_data[0].get("terminate_flag"):
            print("Termination signal received from simulator. Stopping.")
            break

        observations = np.atleast_2d([r["observations"] for r in received_data]).T

        # TELL the optimizer
        tagged_output = Dataset(
            query_points=tf.cast(qp_orig, tf.float64),
            observations=tf.cast(observations, tf.float64),
        )
        bo.tell(tagged_output)

        # Process and save results
        new_sample_df = pd.DataFrame(qp, columns=feature_names)
        for i, col in enumerate(response_cols):
            new_sample_df[col] = observations[:, i]

        # Append to CSV instead of rewriting
        new_sample_df.to_csv(
            csv_path, mode="a", header=not csv_path.exists(), index=False
        )
        results_df = pd.concat([results_df, new_sample_df], ignore_index=True)

        # Update UI
        update_flask_ui(
            post_url, new_sample_df, received_data, bo.models[OBJECTIVE], plot_inputs_2d
        )

    return "Optimization finished."


def main(
    connection_config: dict,
    model_config: dict,
    path_config: dict,
    problem_config: dict,
    default_config_path: str = "config.json",
    verbose: bool = True,
    **kwargs,
) -> None:
    """
    @param connection_config:
    @param model_config:
    @param path_config:
    @param problem_config:
    @param default_config_path:
    @param kwargs:
    @return:
    """
    # --- 1. SETUP ---
    connection_config, model_config, path_config = setup_configs(
        default_config_path=default_config_path,
        connection_config=connection_config,
        model_config=model_config,
        path_config=path_config,
    )

    server_socket, client_socket = setup_networking(connection_config, problem_config)

    problem_name = problem_config.get(
        "problem_name",
        problem_config.get("experiment_parameters", "default").get("problem_name"),
    )
    if problem_name == "default":
        msg = 'No name was provided for the experiment, using "default"'
        warnings.warn(msg)

    problem_type = problem_config.get(
        "problem_type",
        problem_config.get("experiment_parameters", "default").get("problem_type"),
    )
    if problem_type == "default":
        msg = 'No type was provided for the exp[eriment, using "regression" as the default'
        problem_type = "regression"

    csv_path, model_store_path = prepare_experiment_paths(
        save_path=path_config["save_path"], problem_name=problem_name
    )

    post_url = get_post_url(connection_config)

    # TODO
    #  - verify here that the problem can be solved by the simulator (target)
    # i.e. matlab, axonsim, neuron, python, each need to have a simulator file for the given problem.
    #  - verify that the model type can solve the problem type
    # i.e. prevent multiobjective or regression to be solved with classification, etc...

    # --- 2. Initial DATA and MODEL ---
    simulator_config = {
        **problem_config.get("experiment_parameters", {}),
        **{"pulse_parameters": problem_config.get("pulse_parameters", [])},
    }
    search_space, scaler, feat_dict = define_scaler_search_space(
        problem_config=simulator_config, scale_inputs=True
    )

    feature_names = list(feat_dict["variable"].keys())
    first_message = (json.dumps({"Fixed_features": feat_dict["fixed"]}) + "\n").encode()

    if verbose:
        print("Feature names:", feature_names)
        print("First message:", first_message)

    client_socket.sendall(first_message)

    # TODO. Optional. Use other methods for initial sampling.
    qp_orig = search_space.sample_method(
        model_config["init_samples"], sampling_method="sobol"
    ).numpy()
    qp = scaler.inverse_transform(qp_orig) if scaler else qp_orig

    print("feature names:", feature_names)
    qp_json = array_to_list_of_dicts(qp, feature_names)

    if verbose:
        print("First batch")
        print(qp_json)

    response = {
        "message": "first queried points using Sobol method",
        "query_points": qp_json,
    }

    response_json = json.dumps(response) + "\n"
    client_socket.sendall(response_json.encode())

    received_data = fetch_data(client_socket)
    if verbose:
        print("received data:")
        print(received_data)

    if received_data is None:
        raise Exception(
            "No data received from simulator for initial query points... stopping"
        )

    observations = np.atleast_2d([r["observations"] for r in received_data]).T
    results_df = pd.DataFrame(qp, columns=feature_names)
    if observations.shape[1] == 1:
        response_cols = ["response"]
    else:
        response_cols = [f"response_{i}" for i in range(observations.shape[1])]

    for i, col in enumerate(response_cols):
        results_df[col] = observations[:, i]
    results_df.to_csv(csv_path, index=False)

    # --- 3. Build BO Optimizer ---
    # TODO If 'inputFile', load the data and included in the samples.
    # Define whether the results contained are scaled or not (Most likely NOT)
    init_dataset = Dataset(
        query_points=tf.cast(qp_orig, tf.float64),
        observations=tf.cast(observations, tf.float64),
    )

    print("Dataset")
    print(init_dataset)
    if model_config["noise_free"]:
        model_config["kernel_variance"] = 1e-6
    else:
        model_config["kernel_variance"] = None

    model = build_model(
        init_dataset, search_space, model_config, problem_type=problem_type
    )
    print("model")
    print(model)
    # TODO extend to batch_sampling at a later stage
    acq_name = problem_config.get("acquisition", "predictive_variance")

    acquisition_function = select_acquisition_function(
        problem_type=problem_type, acq_name=acq_name
    )

    rule = EfficientGlobalOptimization(
        builder=acquisition_function,
        num_query_points=model_config.get("num_query_points", 1),
    )

    bo = AskTellOptimizerHistory(
        observer=problem_name,
        datasets=init_dataset,
        search_space=search_space,
        models=model,
        acquisition_rule=rule,
        fit_model=True,
        overwrite=True,
        track_path=model_store_path,
    )

    # Prepare grid for 2D contour plot if needed
    plot_inputs_2d = None
    if len(feature_names) == 2:
        # This logic can also be moved to a helper function
        x0_orig = np.linspace(scaler.feature_min[0], scaler.feature_max[0], GRID_POINTS)
        x1_orig = np.linspace(scaler.feature_min[1], scaler.feature_max[1], GRID_POINTS)
        mx_orig, my_orig = np.meshgrid(x0_orig, x1_orig)

        x0_scaled = np.linspace(0, 1, GRID_POINTS)  # Assuming scaled to [0, 1]
        x1_scaled = np.linspace(0, 1, GRID_POINTS)
        mx_scaled, my_scaled = np.meshgrid(x0_scaled, x1_scaled)

        plot_inputs_2d = {
            "original_grid": pd.DataFrame(
                np.column_stack([mx_orig.ravel(), my_orig.ravel()]), columns=["x", "y"]
            ),
            "model_inputs": np.column_stack([mx_scaled.ravel(), my_scaled.ravel()]),
            "shape": mx_scaled.shape,
        }

    update_flask_ui(
        post_url, results_df, received_data, bo.models[OBJECTIVE], plot_inputs_2d
    )

    # TODO, use this timing
    #  start = timeit.default_timer()
    print("Python: Entering the loop ")
    exit_message = run_optimization_loop(
        bo,
        client_socket,
        scaler,
        feature_names,
        response_cols,
        results_df,
        csv_path,
        post_url,
        plot_inputs_2d,
    )

    if verbose:
        print(exit_message)
    cleanup_server_socket(client_sock=client_socket, server_sock=server_socket)


def json_or_file(value: str) -> dict:
    """
    Custom argparse type.
    Tries to parse the value as a JSON string. If that fails, it assumes
    the value is a path to a JSON file and tries to load it.
    """
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        path = Path(value)
        if not path.is_file():
            raise argparse.ArgumentTypeError(
                f"'{value}' is not a valid JSON string or file path."
            )
        with open(path, "r") as f:
            return json.load(f)


def build_configs_from_args(args: argparse.Namespace) -> dict:
    """
    Builds the nested configuration dictionaries from the flat argparse namespace.
    Handles a global config file and individual overrides.
    """
    # Start with a base config from the global file, if provided
    base_config = json_or_file(args.config) if args.config else {}

    # Initialize configs from the base config
    configs = {
        "connection_config": base_config.get("connection_config", {}),
        "model_config": base_config.get("model_config", {}),
        "path_config": base_config.get("path_config", {}),
        "problem_config": base_config.get("problem_config", {}),
        "other_params": {},  # For any args that don't fit the pattern
    }

    # Layer individual config files/strings over the base config.
    if args.connection_config:
        configs["connection_config"].update(args.connection_config)
    if args.model_config:
        configs["model_config"].update(args.model_config)
    if args.path_config:
        configs["path_config"].update(args.path_config)
    if args.problem_config:
        configs["problem_config"].update(args.problem_config)

    # Finally, layer individual command-line overrides from dot-notation args
    args_dict = vars(args)
    for key, value in args_dict.items():
        # We only care about keys with a '.' that were actually provided a value
        if value is None or "." not in key:
            continue

        config_key, param_key = key.split(".", 1)
        config_name = f"{config_key}_config"  # e.g., 'model' -> 'model_config'

        if config_name in configs:
            configs[config_name][param_key] = value
        else:
            # This is a fallback for args that don't match the pattern
            configs["other_params"][key] = value

    return configs


def create_parser() -> argparse.ArgumentParser:
    """Creates and configures the argument parser."""
    parser = argparse.ArgumentParser(
        description="Controller to start Python or MATLAB process."
    )

    # --- Group 1: Configuration via Files ---
    file_group = parser.add_argument_group("Configuration via Files")
    file_group.add_argument(
        "--config",
        type=str,
        help="Path to a global JSON configuration file. Individual arguments will override these values.",
    )
    # The following are for overriding entire sections with a different file
    file_group.add_argument(
        "--connection_config",
        type=json_or_file,
        help="Override connection config with a JSON file or string.",
    )
    file_group.add_argument(
        "--model_config",
        type=json_or_file,
        help="Override model config with a JSON file or string.",
    )
    file_group.add_argument(
        "--path_config",
        type=json_or_file,
        help="Override path config with a JSON file or string.",
    )
    file_group.add_argument(
        "--problem_config",
        type=json_or_file,
        help="Override problem config with a JSON file or string.",
    )

    # --- Group 2: Individual Parameter Overrides ---
    # Use dot notation for clarity. argparse will handle the parsing.
    override_group = parser.add_argument_group("Individual Parameter Overrides")

    # Connection parameters
    override_group.add_argument(
        "--connection.target", choices=["Python", "MATLAB"], help="Target simulator."
    )
    override_group.add_argument(
        "--connection.port", type=int, help="Connection port number."
    )
    override_group.add_argument(
        "--connection.flask_port", type=int, help="Port for the Flask UI."
    )

    # Model parameters
    override_group.add_argument(
        "--model.init_samples", type=int, help="Number of initial samples for BO."
    )
    override_group.add_argument(
        "--model.scale_inputs", action="store_true", help="Flag to scale inputs."
    )

    # # Problem parameters
    # override_group.add_argument('--problem.experiment.name', type=str, help='Name of the experiment.')
    # override_group.add_argument(
    #     '--problem.bounds', type=float, nargs='+',
    #     help='Bounds for a variable, e.g., --problem.bounds 0.1 10.0'
    # )

    return parser


if __name__ == "__main__":
    print("PYTHONPATH:", sys.path)
    parser = create_parser()
    args = parser.parse_args()

    group2_args = [
        args.connection_config,
        args.model_config,
        args.problem_config,
        args.path_config,
    ]
    if not args.config and not any(group2_args) and len(sys.argv) == 1:
        parser.error(
            "No configuration provided. Please provide a --config file or other arguments."
        )

    try:
        all_configs = build_configs_from_args(args)
        main(**all_configs)

    except Exception as e:
        print(f"Error occurred: {e}")
        sys.exit(1)

# api/experiment_runner.py
# Main function that carries the iterative optimization loop.
import socket
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

# Online Neurop imports
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.data import Dataset

# UI and Config imports
from api.backend.plotting import prepare_data_bundle, send_plot_bundle
from api.config_utils import (
    build_configs_from_args,
    get_post_url,
    prepare_experiment_paths,
    setup_configs,
)
from api.frontend.components.config_forms import create_parser
from networking.connection_utils import cleanup_server_socket, setup_networking
from networking.tcp_protocol import receive_message, send_message
from online_neuro.bayessian_optimizer import AskTellOptimizerHistory
from online_neuro.custom_acquisitions import select_acquisition_function
from online_neuro.online_learning import build_model
from online_neuro.utils.data_conversion import array_to_list_of_dicts
from online_neuro.utils.scalers import BaseScaler, define_scaler_search_space

# Development log
# TODO, use timing for logging and additional information.
# TODO If 'inputFile', load the data and included in the samples.
# TODO send stop signals from both ends upon certain conditions (i.e. no improvement, reached goal,
# buget limit, etc...)
# TODO, Add flag. For toy-problems, add the option of generating performance results (i.e. Accuracy/RMSE over
# updates
# TODO. Minot. Allow using other methods for initial sampling.


def run_optimization_loop(
    bo: AskTellOptimizerHistory,
    client_socket: socket.socket,
    scaler: BaseScaler,
    feature_names: list[str],
    response_cols: list[str],
    results_df: pd.DataFrame,
    csv_path: Path,
    post_url: str,
    verbose: bool = False,
    steps: int | None = None,
):
    """
    Execute the main Ask–Tell optimization loop.

    This function repeatedly:
      1. Asks the optimizer for the next query points.
      2. Sends these to the external simulator (via socket).
      3. Receives observations from the simulator.
      4. Updates (tells) the optimizer.
      5. Logs and stores results.
      6. Sends plot update bundles to the Flask UI.

    The loop continues until:
      - `steps` is reached (if provided),
      - the simulator sends a termination flag,
      - the optimizer returns `None`, or
      - the user interrupts with Ctrl+C.

    Parameters
    ----------
    bo : AskTellOptimizerHistory
        The Trieste Ask–Tell optimizer wrapper.

    client_socket :
        Connected socket for communication with the external simulator.

    scaler :
        Optional scaler or transformer providing `inverse_transform()`.
        If None, the raw optimizer inputs are sent directly to the simulator.

    feature_names : list[str]
        Names of the input features corresponding to the optimizer’s search space.

    response_cols : list[str]
        Column names for the returned observations (e.g. objective values).

    results_df : pd.DataFrame
        Accumulated results dataframe, updated in-place by appending new samples.

    csv_path : Path
        Path where CSV results are stored. New rows are appended each step.

    post_url : str
        Base URL of the Flask server for UI updates. If empty, UI updates are skipped.

    verbose : bool, optional
        If True, prints step-wise debug information.

    steps : int | None, optional
        Maximum number of optimization steps.
        If None, optimizes indefinitely until an external stop condition occurs.

    Returns
    -------
    str
        A status message when optimization finishes normally or is interrupted.

    Notes
    -----
    - Communication with the simulator uses `send_message()` and `receive_message()`,
      which are defined in the network utilities
    - UI updates are sent using `send_plot_bundle()`.
    - The function appends to CSV incrementally (streaming-friendly).
    - KeyboardInterrupt is handled gracefully to inform the simulator.
    """

    count = 0
    try:
        while steps is None or count < steps:
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

            # client_socket.sendall((json.dumps(message) + "\n").encode())
            if verbose:
                print("Server - Sending the next qp:", message)

            response = send_message(client_socket, message)

            # RECEIVE observations
            received_data = receive_message(client_socket)

            if isinstance(received_data, dict):
                received_data = [received_data]

            if not received_data or received_data[0].get("terminate_flag"):
                print("Termination signal received from simulator. Stopping.")
                break

            last_observation = received_data[-1]

            if (
                "full_observations" in last_observation
                and "stim_pulse" in last_observation["full_observations"]
                and "block_pulse" in last_observation["full_observations"]
            ):
                pulse_data = {
                    "stim_pulse": last_observation["full_observations"]["stim_pulse"],
                    "block_pulse": last_observation["full_observations"]["block_pulse"],
                    "time": last_observation["full_observations"]["time"],
                }
            else:
                pulse_data = None

            observations = np.atleast_2d([r["observations"] for r in received_data]).T

            # TELL the optimizer
            tagged_output = Dataset(
                query_points=tf.cast(qp_orig, tf.float64),
                observations=tf.cast(observations, tf.float64),
            )

            bo.tell(tagged_output)

            # Process and save results
            new_sample_df = pd.DataFrame(qp, columns=feature_names)
            qp_result = new_sample_df.to_dict(orient="list")
            for i, col in enumerate(response_cols):
                new_sample_df[col] = observations[:, i].tolist()
                qp_result[col] = observations[:, i].tolist()

            qp_result["feature_names"] = feature_names
            qp_result["observation_names"] = response_cols

            # Append to CSV instead of rewriting
            new_sample_df.to_csv(
                csv_path, mode="a", header=not csv_path.exists(), index=False
            )

            results_df = pd.concat([results_df, new_sample_df], ignore_index=True)

            context_dict = {
                "results_df": new_sample_df,
                "bo_model": bo,
                "scaler": scaler,
                "lines_dict": pulse_data,
                "full_observations": last_observation.get("full_observations", None),
            }

            all_data_sources = prepare_data_bundle(
                experiment_name=bo.name, context=context_dict
            )
            send_plot_bundle(
                post_url=post_url,
                experiment_name=bo.name,
                data_sources=all_data_sources,
            )

        if steps == count:
            message = {"terminate_flag": True, "message": "Optimiziation finished"}
            response = send_message(client_socket, message)

    except KeyboardInterrupt:
        if verbose:
            print("\n Interrupted by user, cleaning up...")
            message = {
                "terminate_flag": True,
                "message": "Optimiziation interrupted (keyboard)",
            }
            response = send_message(client_socket, message)
            print(response)

    return "Optimization finished."


def main(
    connection_config: dict,
    model_config: dict,
    path_config: dict,
    problem_config: dict,
    optimizer_config: dict,
    default_config_path: str = "config.json",
    verbose: bool = True,
    **kwargs,
) -> None:
    """
    Run a full Bayesian optimization experiment end-to-end.

    This function orchestrates the entire experiment flow:
      1. Load and merge all configuration dictionaries.
      2. Set up networking with the external simulator.
      3. Initialize the search space, scaler, and fixed/variable features.
      4. Send initial Sobol query batch to the simulator.
      5. Receive observations and construct the initial dataset.
      6. Build the Trieste model and acquisition rule.
      7. Enter the main Ask–Tell optimization loop.
      8. Stream results to CSV, update the UI, and save model checkpoints.
      9. Clean up all networking resources after completion.

    Parameters
    ----------
    connection_config : dict
        Contains networking parameters such as ports, host, and UI connection settings.

    model_config : dict
        Parameters controlling the GP model construction
        (kernel choice, noise handling, likelihood, etc.).

    path_config : dict
        Contains file system paths for saving results, models, and logs.

    problem_config : dict
        Description of the underlying optimization problem including:
        - fixed and variable features,
        - problem type (regression, classification, or MOO),
        - pulse parameters,
        - experiment metadata.

    optimizer_config : dict
        Settings for the optimizer:
        - initial sample count,
        - acquisition function name,
        - number of BO iterations,
        - batch size, etc.

    default_config_path : str, optional
        Optional path to a default JSON configuration file merged with overrides.

    verbose : bool, optional
        If True, prints progress messages throughout the lifecycle.

    **kwargs : Any
        Additional unused keyword arguments for flexibility.

    Returns
    -------
    None
        The function performs an experiment and terminates without returning a value.

    Notes
    -----
    - This function is an orchestration layer and delegates work to helpers such as:
        `setup_configs()`, `setup_networking()`,
        `define_scaler_search_space()`, `build_model()`,
        `run_optimization_loop()`, and `send_plot_bundle()`.
    - Networking failures or missing simulator responses will terminate the experiment.
    - Results are incrementally appended to a CSV file to support long-running experiments.
    """
    # --- 1. SETUP ---
    all_configs = setup_configs(
        default_config_path=default_config_path,
        connection_config=connection_config,
        problem_config=problem_config,
        model_config=model_config,
        path_config=path_config,
        optimizer_config=optimizer_config,
    )
    connection_config = all_configs["connection_config"]
    model_config = all_configs["model_config"]
    path_config = all_configs["path_config"]
    optimizer_config = all_configs["optimizer_config"]
    problem_config = all_configs["problem_config"]

    server_socket, client_socket = setup_networking(connection_config, problem_config)

    problem_name = problem_config["experiment_parameters"]["problem_name"]
    problem_type = problem_config["experiment_parameters"]["problem_type"]

    if problem_type == "default":
        msg = "A problem_type is required, none was specified"
        raise NotImplementedError(msg)
    if problem_name == "default":
        msg = "A problem_name is required, none was specified."
        raise NotImplementedError(msg)

    path_dict = prepare_experiment_paths(
        save_path=path_config["save_path"], problem_name=problem_name
    )
    csv_path = path_dict["csv_path"]
    model_store_path = path_dict["model_path"]

    post_url = get_post_url(connection_config)

    # --- 2. Initial DATA and MODEL ---
    # Colapsing the configuration to a single level
    simulator_config = {
        **problem_config.get("experiment_parameters", {}),
        **{"pulse_parameters": problem_config.get("pulse_parameters", [])},
    }
    search_space, scaler, feat_dict = define_scaler_search_space(
        problem_config=simulator_config, scale_inputs=True
    )

    feature_names = list(feat_dict["variable"].keys())
    first_message = {"Fixed_features": feat_dict["fixed"]}
    if verbose:
        print("Python: Sending Fixed_features message:", first_message)
    send_message(client_socket, first_message)

    # Block and wait
    print("PYTHON: Waiting for client to acknowledge fixed features...")
    feat_ack = receive_message(client_socket)
    if not feat_ack or feat_ack.get("status") != "ready":
        error_msg = f"PYTHON: Did not receive ready signal from CLIENT. Got: {feat_ack}"
        client_socket.close()
        server_socket.close()
        raise ConnectionAbortedError(error_msg)

    # If an initial file was provided, load those samples
    # Define whether the results contained are scaled or not (Most likely NOT)
    # It initial_samples is  lower than the number of required samples,
    #     use random sampling and collect the rest

    qp_orig = search_space.sample_method(
        num_samples=optimizer_config["init_samples"], sampling_method="sobol"
    ).numpy()

    qp = scaler.inverse_transform(qp_orig) if scaler else qp_orig

    qp_json = array_to_list_of_dicts(qp, feature_names)

    if verbose:
        print("First batch")
        print(qp_json)

    response = {
        "message": "first queried points using Sobol method",
        "query_points": qp_json,
    }

    response = send_message(client_socket, response)
    received_data = receive_message(client_socket)

    if received_data is None:
        raise Exception(
            "No data received from simulator for initial query points... stopping"
        )
    if isinstance(received_data, dict):
        received_data = [received_data]

    last_observation = received_data[-1]

    if (
        "full_observations" in last_observation
        and "stim_pulse" in last_observation["full_observations"]
        and "block_pulse" in last_observation["full_observations"]
    ):
        pulse_data = {
            "stim_pulse": last_observation["full_observations"]["stim_pulse"],
            "block_pulse": last_observation["full_observations"]["block_pulse"],
            "time": last_observation["full_observations"]["time"],
        }
    else:
        pulse_data = None

    observations = np.atleast_2d([r["observations"] for r in received_data]).T
    results_df = pd.DataFrame(qp, columns=feature_names)
    if observations.shape[1] == 1:
        response_cols = ["response"]
    else:
        response_cols = [f"response_{i}" for i in range(observations.shape[1])]

    for i, col in enumerate(response_cols):
        results_df[col] = observations[:, i].tolist()
    results_df.to_csv(csv_path, index=False)

    if verbose:
        print("First results saved to: ", csv_path)

    qp_result = results_df.to_dict(orient="list")
    qp_result["feature_names"] = feature_names
    qp_result["observation_names"] = response_cols

    # --- 3. Build BO Optimizer ---
    init_dataset = Dataset(
        query_points=tf.cast(qp_orig, tf.float64),
        observations=tf.cast(observations, tf.float64),
    )

    if model_config["noise_free"]:
        model_config["kernel_variance"] = 1e-6
    else:
        model_config["kernel_variance"] = None

    model = build_model(
        init_dataset, search_space, model_config, problem_type=problem_type
    )
    if verbose:
        print("model")
        print(model)

    acq_name = optimizer_config.get("acquisition", None)
    if acq_name is None:
        raise NotImplementedError(
            "Acquisition functions need to be defined, defaulting is not allowed"
        )

    acquisition_function = select_acquisition_function(
        problem_type=problem_type, acq_name=acq_name
    )

    rule = EfficientGlobalOptimization(
        builder=acquisition_function,
        num_query_points=optimizer_config.get("num_query_points", 1),
    )

    bo = AskTellOptimizerHistory(
        name=problem_name,
        datasets=init_dataset,
        search_space=search_space,
        models=model,
        acquisition_rule=rule,
        fit_model=True,
        overwrite=True,
        track_path=model_store_path,
    )

    context_dict = {
        "results_df": results_df,
        "bo_model": bo,
        "scaler": scaler,
        "lines_dict": pulse_data,
        "full_observations": last_observation.get("full_observations", None),
    }

    all_data_sources = prepare_data_bundle(
        experiment_name=bo.name, context=context_dict, with_meta=True
    )

    send_plot_bundle(
        post_url=post_url,
        experiment_name=problem_name,
        data_sources=all_data_sources,
    )

    #  start = timeit.default_timer()
    if verbose:
        print("Python: Entering the loop ")

    exit_message = run_optimization_loop(
        bo=bo,
        client_socket=client_socket,
        scaler=scaler,
        feature_names=feature_names,
        response_cols=response_cols,
        results_df=results_df,
        csv_path=csv_path,
        post_url=post_url,
        steps=optimizer_config["n_iters"],
        verbose=verbose,
    )

    if verbose:
        print(exit_message)
        print("Optimization loop succesfully completed, cleaning up...")

    cleanup_server_socket(client_sock=client_socket, server_sock=server_socket)


if __name__ == "__main__":
    print("PYTHONPATH:", sys.path)
    print("Experiment runner ")
    parser = create_parser()
    args = parser.parse_args()

    group2_args = [
        args.connection_config,
        args.model_config,
        args.problem_config,
        args.path_config,
        args.optimizer_config,
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

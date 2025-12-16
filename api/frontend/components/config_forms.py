# api/frontend/components/config_forms.py
"""
Module: config_forms
-------------------

Provides mappings and functions for experiment, model, acquisition, and pulse
configurations. These configurations are loaded from JSON files and used to
dynamically generate experiment parameters, model parameters, and plots.

Key components:

- template_map: Maps experiment identifiers to JSON template files.
- model_map: Maps model identifiers to model JSON files.
- acquisition_map: Maps acquisition function names to JSON configuration files.
- add_model_map: Additional shared JSON files for models.
- function_map: Maps pulse function names to JSON parameter files.
- optimizer_config_path: Global path to optimizer configuration.
- matlab_experiments: User-facing names mapped to internal experiment IDs (MATLAB).
- python_experiments: User-facing names mapped to internal experiment IDs (Python).
- experiments_types: Mapping of experiment IDs to type: classification, regression, or moo.
- plot_configurations: Mapping of experiment IDs to a list of plot dicts.
"""

import argparse
from pathlib import Path
from typing import Any, Literal

from api.config_utils import json_or_file
from common import utils as nn_utils

PlotType = Literal["scatter", "line", "contour", "lines"]
PlotParams = dict[str, str]
PlotConfig = dict[str, object]

# Mapping experiment template names to JSON paths
template_map: dict[str, Path] = {
    "axonsim_nerve_block": Path("config") / "experiments" / "axonsim_template.json",
    "axonsim_regression": Path("config") / "experiments" / "axonsim_template.json",
    "matlab_rose_regression": Path("config") / "experiments" / "rose_template.json",
    "circle_classification": Path("config") / "experiments" / "circle_template.json",
    "vlmop2": Path("config") / "experiments" / "vlmop2_template.json",
    "cajal_ap_block": Path("config") / "experiments" / "cajal_template.json",
}

# Mapping model names to JSON files
model_map: dict[str, Path] = {
    "mc_dropout_nn": Path("config") / "models" / "mc_nn.json",
    "ensemble_nn": Path("config") / "models" / "ensemble_nn.json",
    "gp": Path("config") / "models" / "gp.json",
}

# Mapping acquisition function names to JSON files
acquisition_map: dict[str, Path] = {
    "negative_predictive_mean": Path("config")
    / "acquisitions"
    / "negative_predictive_mean.json",
    "minimum_variance": Path("config") / "acquisitions" / "minimum_variance.json",
    "bald": Path("config") / "acquisitions" / "bald.json",
}

# Additional shared JSON model configs
add_model_map: dict[str, Path] = {
    "nn": Path("config") / "models" / "nn.json",
    "common": Path("config") / "models" / "common.json",
    "common_nn": Path("config") / "models" / "common_nn.json",
}

# Pulse functions configuration mapping
function_map: dict[str, Path] = {
    "pulse_ramp": Path("config") / "custom_pulses" / "pulse_ramp.json",
    "monophasic": Path("config") / "custom_pulses" / "monophasic.json",
    "default": Path("config") / "custom_pulses" / "axonsim_default.json",
}

# Optimizer configuration file
optimizer_config_path: Path = Path("config") / "optimization" / "common.json"

# Mapping of user-friendly experiment names to internal IDs (MATLAB)
matlab_experiments: dict[str, str] = {
    "Axonsim (nerve block)": "axonsim_nerve_block",
    "Axonsim (regression)": "axonsim_regression",
    "Matlab toy regression": "matlab_rose_regression",
    "Toy Classification": "circle_classification",
    "Toy VLMOP2": "vlmop2",
}

# Mapping of user-friendly experiment names to internal IDs (Python)
python_experiments: dict[str, str] = {
    "Toy classification (Python)": "circle_classification",
    # "Toy regression (Python)": "matlab_rose_regression",
    "Toy MOO (Python)": "vlmop2",
    "Cajal nerve block": "cajal_ap_block",
}

# Type of each experiment by internal ID
experiments_types: dict[str, Literal["classification", "regression", "moo"]] = {
    "axonsim_nerve_block": "classification",
    "axonsim_regression": "regression",
    "cajal_ap_block": "classification",
    "matlab_rose_regression": "regression",
    "circle_classification": "classification",
    "vlmop2": "moo",
}

# Mapping experiment IDs to lists of plot configurations
plot_configurations: dict[str, list[PlotConfig]] = {
    "cajal_ap_block": [
        {
            "id": "1",
            "type": "lines",
            "src": "pulse_data",
            "generator": "plot_line",
            "params": {
                "y_cols": ["stim_pulse", "block_pulse"],
                "x_col": "time",
                "title": "Stimulation and Blocking Pulses",
                "xaxis_title": "Time (ms)",
                "yaxis_title": "Amplitude (mA)",
            },
        },
        {
            "id": "2",
            "type": "scatter",
            "src": "scatter_input_input",
            "generator": "scatter_from_dataframe",
            "params": {
                "x_col": "pulse_parameters_1_s_pw",
                "y_col": "pulse_parameters_1_s_amp",
                "color_col": "response",
                "xaxis_title": "pulse_1_pw (ms)",
                "yaxis_title": "pulse_1_amp (mA)",
            },
        },
        {
            "id": "3",
            "type": "contour",
            "src": "cajal_data",
            "generator": "contour_from_cajal_results",
            "params": {
                "zmin": -80,
                "zmax": 50,
                "title": "AP Propagation",
                "xaxis_title": "Time (ms)",
                "yaxis_title": "Node (index)",
            },
        },
    ],
    "axonsim_nerve_block": [
        {
            "id": "1",
            "type": "scatter",
            "src": "scatter_input_input",
            "generator": "scatter_from_dataframe",
            "params": {
                "y_col": "pulse_parameters_1_s_amp",
                "x_col": "pulse_parameters_1_s_pw",
                "color_col": "response",
                "xaxis_title": "Time (ms)",
                "yaxis_title": "Amplitude (mA)",
            },
        },
        {
            "id": "2",
            "type": "scatter",
            "src": "scatter_input_input",
            "generator": "scatter_from_dataframe",
            "params": {
                "y_col": "pulse_parameters_1_s_pw",
                "x_col": "pulse_parameters_1_s_delay",
                "color_col": "response",
                "xaxis_title": "Pulse 1 width (ms)",
                "yaxis_title": "Delay 1 (ms)",
            },
        },
        {
            "id": "3",
            "type": "contour",
            "src": "model_uncertainty",
            "generator": "contour_from_model",
            "params": {
                "x_col": "pulse_parameters_1_s_amp",
                "y_col": "pulse_parameters_1_s_pw",
                "z_source": "mean",
                "xaxis_title": "Pulse 1 amp (mA)",
                "yaxis_title": "Pulse 1 width (ms)",
            },
        },
    ],
    "circle_classification": [
        {
            "id": "0",
            "type": "scatter",
            "src": "scatter_input_input",
            "generator": "scatter_from_dataframe",
            "params": {
                "x_col": "x0",
                "y_col": "x1",
                "color_col": "response",
                "xaxis_title": "x0",
                "yaxis_title": "x1",
            },
        },
        {
            "id": "1",
            "type": "contour",
            "src": "model_uncertainty",
            "generator": "contour_from_model",
            "params": {
                "x_col": "x0",
                "y_col": "x1",
                "z_source": "mean",
                "xaxis_title": "x0",
                "yaxis_title": "x1",
            },
        },
    ],
    "axonsim_regression": [
        {
            "id": "0",
            "type": "scatter",
            "src": "scatter_input_input",
            "generator": "scatter_from_dataframe",
            "params": {
                "x_col": "pulse_parameters_1_s_amp",
                "y_col": "pulse_parameters_1_s_pw",
                "xaxis_title": "pulse_1_amp (mA)",
                "yaxis_title": "pulse_1_pw (ms)",
            },
        },
        {
            "id": "1",
            "type": "contour",
            "src": "model_uncertainty",
            "generator": "contour_from_model",
            "params": {
                "x_col": "pulse_parameters_1_s_amp",
                "y_col": "pulse_parameters_1_s_pw",
                "z_source": "mean",
                "xaxis_title": "pulse_1_amp (mA)",
                "yaxis_title": "pulse_1_pw (ms)",
            },
        },
    ],
    "matlab_rose_regression": [
        {
            "id": "0",
            "type": "scatter",
            "src": "scatter_input_input",
            "generator": "scatter_from_dataframe",
            "params": {
                "x_col": "x0",
                "y_col": "x1",
                "color_col": "response",
                "xaxis_title": "x0",
                "yaxis_title": "x1",
            },
        },
        {
            "id": "1",
            "type": "contour",
            "src": "model_uncertainty",
            "generator": "contour_from_model",
            "params": {
                "x_col": "x0",
                "y_col": "x1",
                "z_source": "mean",
                "xaxis_title": "x0",
                "yaxis_title": "x1",
            },
        },
    ],
    "vlmop2": [
        {
            "id": "0",
            "type": "scatter",
            "src": "scatter_input_input",
            "generator": "scatter_from_dataframe",
            "params": {
                "x_col": "x0",
                "y_col": "x1",
                "color_col": "response_0",
                "xaxis_title": "x0",
                "yaxis_title": "x1",
            },
        },
        {
            "id": "1",
            "type": "line",
            "src": "objective",
            "generator": "scatter_from_dataframe",
            "params": {
                "x_col": "response_0",
                "y_col": "response_1",
                "xaxis_title": "Objective 1",
                "yaxis_title": "Objective 2",
            },
        },
        {
            "id": "2",
            "type": "line",
            "src": "objective",
            "generator": "plot_line",
            "params": {
                "x_col": "iteration",
                "y_col": "response_0",
                "title": "Best Objectives over Iterations",
                "xaxis_title": "Iteration",
                "yaxis_title": "Objective Value",
            },
        },
        {
            "id": "3",
            "type": "line",
            "src": "objective",
            "generator": "plot_line",
            "params": {
                "x_col": "iteration",
                "y_col": "response_1",
                "title": "Best Objectives over Iterations",
                "xaxis_title": "Iteration",
                "yaxis_title": "Objective Value",
            },
        },
    ],
}


def verify_optim_fixed(config: dict) -> None:
    """
    Validate that configuration dictionaries do not have conflicting
    'optimizable' and 'user_fixed' flags.

    The rule enforced:
        - If both keys exist, they must not be simultaneously True.
        - If both are False, it's allowed.

    Parameters
    ----------
    config : dict
        Configuration dictionary to validate. Can contain nested dictionaries.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If an invalid combination of 'optimizable' and 'user_fixed' is found.
    """
    for k, v in config.items():
        if isinstance(v, dict):
            if ("optimizable" in v) and ("user_fixed" in v):
                assert (v["optimizable"] != v["user_fixed"]) or (
                    v["optimizable"] is False and v["user_fixed"] is False
                )


def config_function(function: str) -> dict[str, Any]:
    """
    Load configuration for a specific pulse function.

    If the function name is unknown, the 'default' function config is used.

    Parameters
    ----------
    function : str
        Name of the function (case-insensitive) for which to load parameters.

    Returns
    -------
    dict
        Configuration dictionary loaded from the corresponding JSON file.

    Raises
    ------
    AssertionError
        If 'verify_optim_fixed' fails on the loaded config.
    """
    function = function.lower()
    if function in function_map:
        config = nn_utils.load_json(function_map[function])
    else:
        config = nn_utils.load_json(function_map["default"])

    verify_optim_fixed(config)
    return config


def config_problem(problem: str) -> dict[str, Any]:
    """
    Load experiment/problem configuration from a JSON template.

    Dynamically prepares a dictionary for generating input widgets based
    on the loaded configuration keys.

    Parameters
    ----------
    problem : str
        Name of the experiment/problem (case-insensitive) to load.

    Returns
    -------
    dict
        Configuration dictionary with an added 'experiment' entry:
        {'experiment': {'value': <problem_name>}}.

    Raises
    ------
    NotImplementedError
        If no configuration file exists for the given problem.
    AssertionError
        If 'verify_optim_fixed' fails on the loaded config.
    """
    problem = problem.lower()
    try:
        config = nn_utils.load_json(template_map[problem])
        verify_optim_fixed(config)

        config["experiment"] = dict()
        config["experiment"]["value"] = problem

    except NotImplementedError as e:
        raise NotImplementedError(
            f"No configuration file for experiment: {problem}"
        ) from e

    return config


def config_model(model: str) -> dict[str, Any]:
    """
    Load configuration for a model, merging it with common and NN-specific configurations.

    Checks for conflicting keys and raises errors if found.

    Parameters
    ----------
    model : str
        Name of the model to load (case-insensitive). Must exist in `model_map`.

    Returns
    -------
    dict
        Merged configuration dictionary containing model-specific and common parameters.

    Raises
    ------
    ValueError
        If the model is unknown or a JSON file cannot be loaded.
    KeyError
        If there are conflicting keys between common or NN common config and model config.
    """
    model = model.lower()
    if model not in model_map:
        raise ValueError(f"Unknown model: {model}")
    try:
        config = nn_utils.load_json(model_map[model])
    except NotImplementedError as e:
        raise ValueError(f"Could not load model config ({model_map[model]}): {e}")

    try:
        common = nn_utils.load_json(add_model_map["common"])
    except ValueError as e:
        raise ValueError(
            f"Could not load common config ({add_model_map['common']}): {e}"
        )

    for k, v in common.items():
        if k in config:
            raise KeyError(
                f"Conflict: key '{k}' exists in both common and {model} configuration."
            )
        config[k] = v

    if model in {"mc_dropout_nn", "ensemble_nn"}:
        try:
            nn_common = nn_utils.load_json(add_model_map["common_nn"])
        except Exception as e:
            raise ValueError(
                f"Could not load NN common config ({add_model_map['common_nn']}): {e}"
            )

        for k, v in nn_common.items():
            if k in config:
                raise KeyError(
                    f"Conflict: key '{k}' in common_nn also present in {model} configuration."
                )
            config[k] = v

    return config


def config_optimizer() -> dict[str, Any]:
    """
    Load global optimizer configuration.

    Returns
    -------
    dict
        Configuration dictionary for the optimizer.

    Raises
    ------
    NotImplementedError
        If the optimizer configuration file cannot be loaded.
    """
    try:
        config = nn_utils.load_json(optimizer_config_path)
    except Exception as e:
        msg = f"Failed at trying to load the optimizar configuration file: {e}"
        raise NotImplementedError(msg)

    return config


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
    file_group.add_argument(
        "--optimizer_config",
        type=json_or_file,
        help="Override optimizer config with a JSON file or string.",
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

    return parser

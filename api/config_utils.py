# api/config_utils.py
import argparse
import json
from datetime import datetime as dt
from pathlib import Path
from typing import Any

from common.utils import load_json


def setup_configs(
    default_config_path: str,
    connection_config: dict,
    problem_config: dict,
    model_config: dict,
    optimizer_config: dict,
    path_config: dict,
) -> dict[str, dict]:
    """
    Load the default configuration file and merge it with runtime overrides.

    This function reads the JSON configuration located at ``default_config_path``
    and merges its sections with the user-provided dictionaries. User-provided
    values always take precedence over defaults.

    In general, as the project has grown, the default values have become placeholders,
    and should not be used in most cases.

    Parameters
    ----------
    default_config_path : str
        Path to the JSON file containing default configuration values.
    connection_config : dict
        Overrides for the connection-related configuration block.
    model_config : dict
        Overrides for the model configuration block.
    optimizer_config : dict
        Overrides for the optimizer configuration block.
    path_config : dict
        Overrides for the path-related configuration block.

    Returns
    -------
    dict[str, dict]
        A dictionary containing the merged configuration sections:
        ``{"connection_config", "model_config", "optimizer_config", "path_config"}``.

    Raises
    ------
    AssertionError
        If the loaded JSON file does not contain a dictionary.
    """

    default_dict = load_json(default_config_path)
    assert isinstance(
        default_dict, dict
    ), "Configuration file is not a valid dictionary"

    connection_config = {
        **default_dict.get("connection_config", {}),
        **connection_config,
    }
    model_config = {**default_dict.get("model_config", {}), **model_config}
    optimizer_config = {**default_dict.get("optimizer_config", {}), **optimizer_config}
    problem_config = {**default_dict.get("problem_config", {}), **problem_config}
    path_config = {**default_dict.get("path_config", {}), **path_config}

    return {
        "connection_config": connection_config,
        "problem_config": problem_config,
        "model_config": model_config,
        "optimizer_config": optimizer_config,
        "path_config": path_config,
    }


def prepare_experiment_paths(
    save_path: str | Path, problem_name: str | Path
) -> dict[str, Path]:
    """
    Create and return the directory structure for storing experiment outputs.

    A folder is created under ``save_path/problem_name`` along with a ``models``
    subdirectory. The returned paths include locations for results, models, and
    a metadata JSON file with a timestamp.

    Parameters
    ----------
    save_path : str | Path
        The base directory where experiment results should be stored.
    problem_name : str | Path
        Name of the experiment or problem; used as a subdirectory.

    Returns
    -------
    dict[str, Path]
        A dictionary with the following keys:
        - ``"root"``: Root directory for this experiment.
        - ``"csv_path"``: Path to the results CSV file.
        - ``"model_path"``: Directory where models will be saved.
        - ``"json_path"``: Path to the timestamped metadata JSON file.

    Notes
    -----
    All folders are created if they do not already exist.
    """
    save_path = Path(save_path) / problem_name
    model_store_path = save_path / "models"
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")

    csv_path = save_path / "results.csv"
    json_path = save_path / f"metadata_{timestamp}.json"

    save_path.mkdir(parents=True, exist_ok=True)
    model_store_path.mkdir(parents=True, exist_ok=True)

    return {
        "root": save_path,
        "csv_path": csv_path,
        "model_path": model_store_path,
        "json_path": json_path,
    }


def get_post_url(connection_config: dict[str, Any]) -> str:
    """
    Return the base POST URL for the Flask server specified in the connection
    configuration.

    Parameters
    ----------
    connection_config : dict
        Configuration containing optional key ``"port_flask"`` with an integer
        port number for the Flask server.

    Returns
    -------
    str
        A string of the form ``"http://localhost:<port>"`` if a port is provided.
        Returns an empty string if no port is found.
    """
    port_flask = connection_config.get("port_flask")
    return f"http://localhost:{port_flask}" if port_flask else ""


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
        "optimizer_config": base_config.get("optimizer_config", {}),
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
    if args.optimizer_config:
        configs["optimizer_config"].update(args.optimizer_config)

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

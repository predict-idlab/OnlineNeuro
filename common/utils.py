# common/utils.py
import json
import os
import sys
from pathlib import Path
from typing import Any


def load_json(json_file: str | Path) -> dict[str, Any]:
    """
    Loads a JSON file from a given path, enforcing that the root element is a dictionary.

    Parameters
    ----------
    json_file : str or pathlib.Path
        Path to the JSON file to be loaded.

    Returns
    -------
    dict[str, Any]
        The content of the JSON file as a dictionary.

    Raises
    ------
    ValueError
        If the root element of the loaded JSON file is not a dictionary (object).
    json.JSONDecodeError
        If the file contains invalid JSON syntax.
    FileNotFoundError
        If the file does not exist.
    """
    with open(json_file) as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Configuration JSON files must be dictionaries/objects")

    return data


def configure_python_paths():
    """
    Adds core project directories and optional external paths from config.json to sys.path.

    This function should be called at the start of scripts or notebooks before
    importing local project modules, ensuring all necessary custom library paths
    are available.

    It performs the following actions:
    1. Determines the project's base directory (two levels up from the calling file).
    2. Checks for a 'config.json' file in the base directory.
    3. If 'config.json' exists and contains a 'path_config' section, it attempts
       to load and add paths specified under keys like 'axonsim_path', 'cajal_path',
       and 'axonml_path' to `sys.path` if they are valid directories and not already present.
    """
    # Get base directory (assumes app.py is in app/)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print(f"Base directory: {base_dir}")

    # Check for config.json and add any valid custom paths
    config_file = os.path.join(base_dir, "config.json")
    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
            path_config = config.get("path_config", {})
            path_keys = ["axonsim_path", "cajal_path", "axonml_path"]

            for key in path_keys:
                custom_path = path_config.get(key)
                if (
                    custom_path
                    and os.path.isdir(custom_path)
                    and custom_path not in sys.path
                ):
                    sys.path.append(custom_path)
        except Exception as e:
            print(f"Warning: Failed to read config.json: {e}")

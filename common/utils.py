# common/utils.py
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Union


def load_json(json_file: Union[str, Path]) -> Union[Dict[str, Any], List[Any]]:
    """
    @param json_file: path to the json file
    @return: loaded json, either dict or list.
    """
    with open(json_file) as f:
        return json.load(f)


def configure_python_paths():
    """
    Adds core project directories and optional external paths from config.json to sys.path.
    Should be called before importing local project modules.
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
            for key in ["axonsim_path", "cajal_path", "axonml_path"]:
                custom_path = path_config.get(key)
                if (
                    custom_path
                    and os.path.isdir(custom_path)
                    and custom_path not in sys.path
                ):
                    sys.path.append(custom_path)
        except Exception as e:
            print(f"Warning: Failed to read config.json: {e}")

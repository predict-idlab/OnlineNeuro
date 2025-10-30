# api/config_utils.py
from pathlib import Path

from common.utils import load_json


def setup_configs(
    default_config_path: str,
    connection_config: dict,
    model_config: dict,
    path_config: dict,
) -> tuple[dict, dict, dict]:
    """Loads default configuration and merges it with provided configs."""
    default_dict = load_json(default_config_path)
    assert isinstance(
        default_dict, dict
    ), "Configuration file is not a valid dictionary"

    connection_config = {
        **default_dict.get("connection_config", {}),
        **connection_config,
    }
    model_config = {**default_dict.get("model_config", {}), **model_config}
    path_config = {**default_dict.get("path_config", {}), **path_config}

    return connection_config, model_config, path_config


def prepare_experiment_paths(
    save_path: str | Path, problem_name: str | Path
) -> tuple[Path, Path]:
    """Creates and returns paths for saving results and models."""
    save_path = Path(save_path) / problem_name
    csv_path = save_path / "results.csv"
    model_store_path = save_path / "models"

    save_path.mkdir(parents=True, exist_ok=True)
    model_store_path.mkdir(parents=True, exist_ok=True)

    return csv_path, model_store_path

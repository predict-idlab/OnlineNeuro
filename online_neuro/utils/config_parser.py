# online_neuro/utils/config_parser.py
from typing import Any


def _parse_parameter(
    base_name: str,
    config: Any,
    feature_info: dict[str, dict],
    lower_bounds: list[float],
    upper_bounds: list[float],
):
    """
    Recursively parse a configuration structure to categorize parameters.

    This function traverses a nested configuration (composed of dictionaries, lists,
    and primitive values) and populates the `feature_info`, `lower_bounds`, and
    `upper_bounds` arguments in-place.

    The function identifies parameters based on specific dictionary schemas:
    - Fixed values: A literal value (e.g., 5, "hello") or a dict `{'value': ...}`.
    - Categorical values: A dict `{'choices': [...]}`.
    - Variable (numeric) values: A dict `{'min_value': ..., 'max_value': ...}`.
      - If min and max are equal, it's treated as a fixed parameter.
      - If they are lists, it's treated as a vector parameter.

    Naming Conventions for Nested Structures:
    - Nested dicts: The parent key is discarded (e.g., `{'a': {'b': 1}}` produces a
      parameter named `b`).
    - Lists of dicts: The base name is enumerated (e.g., `{'a': [{'v':1}]}` produces
      a parameter named `a_0`).
    - Single-item vectors: For a variable parameter where `min_value` and
      `max_value` are lists of length one, the name is not enumerated
      (e.g., `param` instead of `param_0`).

    Args:
        base_name: The current name of the parameter being processed.
        config: The configuration value or structure for the given base_name.
        feature_info: A dictionary to be populated with categorized feature
                      information. It is modified in-place.
        lower_bounds: A list to be populated with lower bounds of variable
                      parameters. It is modified in-place.
        upper_bounds: A list to be populated with upper bounds of variable
                      parameters. It is modified in-place.
    """

    if isinstance(config, dict):
        if "choices" in config:
            feature_info["categorical"][base_name] = config["choices"]

        elif "min_value" in config and "max_value" in config:
            min_val, max_val = config["min_value"], config["max_value"]

            if isinstance(min_val, list):
                if not isinstance(max_val, list) or len(min_val) != len(max_val):
                    raise ValueError(
                        f"Parameter '{base_name}': 'min_value' and 'max_value' must be lists of the same length."
                    )

                is_single_item_vector = len(min_val) == 1

                for i, (min_i, max_i) in enumerate(zip(min_val, max_val)):
                    # If it's a single item, use the base_name directly. Otherwise, enumerate.
                    feat_name = (
                        base_name if is_single_item_vector else f"{base_name}_{i}"
                    )

                    if min_i == max_i:
                        feature_info["fixed"][feat_name] = min_i
                    else:
                        feature_info["variable"][feat_name] = None
                        lower_bounds.append(min_i)
                        upper_bounds.append(max_i)

            # Scalar numeric values
            else:
                if min_val == max_val:
                    feature_info["fixed"][base_name] = min_val
                else:
                    feature_info["variable"][base_name] = None
                    lower_bounds.append(min_val)
                    upper_bounds.append(max_val)

        elif "value" in config:
            feature_info["fixed"][base_name] = config["value"]

        else:
            # Assumes it's a container. Recurse into its items.
            for sub_name, sub_config in config.items():
                nested_name = f"{base_name}_{sub_name}"
                _parse_parameter(
                    nested_name, sub_config, feature_info, lower_bounds, upper_bounds
                )

    elif isinstance(config, list) and all(isinstance(item, dict) for item in config):
        for i, item_config in enumerate(config):
            nested_name = f"{base_name}_{i}"
            _parse_parameter(
                nested_name, item_config, feature_info, lower_bounds, upper_bounds
            )

    else:
        # Any other type is a fixed value.
        feature_info["fixed"][base_name] = config

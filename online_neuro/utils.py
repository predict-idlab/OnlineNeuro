# /online_neuro/utils.py
import json
import queue
import socket
import subprocess
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
from trieste.space import Box
from trieste.types import TensorType

from .matlab_utils import run_matlab

__all__ = ["run_matlab"]


class CustomBox(Box):
    """Superclass of trieste.spae.Box to handle different sampling methods
       using a str argument "sampling_method".

    Args:
        Box (trieste.space.Box): Trieste's native Box class for search space
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample_method(
        self,
        num_samples: int,
        seed: Optional[int] = None,
        skip: Optional[int] = None,
        max_tries: int = 100,
        sampling_method: str = "sobol",
    ) -> TensorType:
        if sampling_method == "random":
            x = self.sample(num_samples, seed)
        elif sampling_method == "halton":
            x = self.sample_halton(num_samples, seed)
        elif sampling_method == "sobol":
            x = self.sample_sobol(num_samples, skip)
        elif sampling_method == "random_feasible":
            x = self.sample_feasible(num_samples, seed, max_tries)
        elif sampling_method == "halton_feasible":
            x = self.sample_feasible(num_samples, seed, max_tries)
        elif sampling_method == "sobol_feasible":
            x = self.sample_feasible(num_samples, skip, max_tries)
        else:
            raise ValueError(f"Unsupported sampling method: {sampling_method}")

        assert (
            x is not None
        ), f"Sampled method {sampling_method} did not return valid samples"
        return x


@DeprecationWarning
class SearchSpacePipeline:
    # Class got discontinued in favor of separated classes for SearchSpace and Scaler
    def __init__(
        self, search_space, mapping: dict, feature_names: list | np.ndarray, scaler=None
    ):
        """
        @param search_space:
        @param mapping:
        @param feature_names:
        @param scaler:
        """
        self.search_space = search_space
        self.mapping = mapping
        self._feature_names = feature_names
        self.scaler = scaler

    def sample(self, num_samples, sample_method="sobol"):
        if sample_method == "sobol":
            samples = self.search_space.sample_sobol(num_samples)
        else:
            raise NotImplementedError(
                f"Sampling method {sample_method} not implemented (yet)."
            )

        samples = self.inverse_transform(samples.numpy())
        return samples

    def inverse_transform(self, x, with_array=True):
        """
        @param x:
        @param with_array:
        @return:
        """
        if self.scaler:
            x = self.scaler.inverse_transform(x)

        as_list = []
        for sample_idx in range(x.shape[0]):
            sample_dict = {k: None for k in self.feature_names()}
            for k in self.feature_names():
                sample_dict[k] = x[sample_idx, self.mapping[k]]

            as_list.append(sample_dict)

        if with_array:
            return as_list, x
        else:
            return as_list

    def feature_names(self):
        return self._feature_names

    def feature_names_ixs(self):
        feature_names_ixs = []
        for fn in self.feature_names():
            feat_ixs_list = [f"{fn}_{ix}" for ix in range(len(self.mapping[fn]))]
            feature_names_ixs.extend(feat_ixs_list)

        return feature_names_ixs


class BaseScaler:
    """
    A base class defining the interface for a scaler.

    This class serves as a template for scalers, ensuring that any subclass
    will have the required transformation methods. It is designed to satisfy
    static analysis tools like Pylance by defining a consistent API.

    Subclasses must implement the 'transform' and 'inverse_transform' methods.
    """

    def __init__(self, feature_min, feature_max, output_range=(0, 1)):
        """
        Initialize the scaler with min and max bounds for each feature.
        This base implementation does not store or process these values.
        """
        pass

    def transform(self, x):
        """
        Transform the input data to the specified output range.
        This method must be overridden by a subclass.
        """
        raise NotImplementedError("Subclasses must implement the 'transform' method.")

    def inverse_transform(self, x):
        """
        Inverse transform the scaled data back to the original range.
        This method must be overridden by a subclass.
        """
        raise NotImplementedError(
            "Subclasses must implement the 'inverse_transform' method."
        )

    def inverse_transform_ix(self, x, ix):
        """
        Inverse transform a single feature of the scaled data.
        This method must be overridden by a subclass.
        """
        raise NotImplementedError(
            "Subclasses must implement the 'inverse_transform_ix' method."
        )

    def inverse_transform_mat(self, x, ix):
        """
        Inverse transform an array using the scaling parameters of a single feature.
        This method must be overridden by a subclass.
        """
        raise NotImplementedError(
            "Subclasses must implement the 'inverse_transform_mat' method."
        )


class IdentityScaler(BaseScaler):
    """
    A scaler that performs no operation (the identity transformation).

    This class conforms to the scaler interface but returns the input
    data unchanged. It is useful as a placeholder when scaling is
    not required, avoiding the need for conditional checks for None.
    """

    def __init__(self, *args, **kwargs):
        """Accepts any arguments to match the real scaler's signature, but ignores them."""
        pass

    def transform(self, x):
        """Returns the input data unchanged."""
        return x

    def inverse_transform(self, x):
        """Returns the input data unchanged."""
        return x

    def inverse_transform_ix(self, x, ix):
        """Returns the input data unchanged."""
        return x

    def inverse_transform_mat(self, x, ix):
        """Returns the input data unchanged."""
        return x


class CustomMinMaxScaler(BaseScaler):
    def __init__(self, feature_min, feature_max, output_range=(0, 1)):
        """
        Initialize the scaler with min and max bounds for each feature.

        Parameters:
        - feature_min: Array-like of minimum values for each feature.
        - feature_max: Array-like of maximum values for each feature.
        - output_range: Range to scale outputs.
        @note This function does not raise warnings if future samples are above or bellow
               the expected min, maxs

        # Example usage
        feature_min = [0, 10, -1]  # Min bounds for each feature
        feature_max = [100, 50, 1]  # Max bounds for each feature

        scaler = customMinMaxScaler(feature_min, feature_max, output_range=(-1, 1))

        X = np.array([[0, 10, -1], [50, 30, 0], [75, 40, -0.5], [100, 50, 0.9999], [-1, -1, -2]])
        X_scaled = scaler.transform(X)
        print("Scaled Data:\n", X_scaled)

        X_original = scaler.inverse_transform(X_scaled)
        print("Original Data:\n", X_original)

        """

        self.num_features = len(feature_min)
        self.feature_min = np.array(feature_min)
        self.feature_max = np.array(feature_max)
        self.feature_range = self.feature_max - self.feature_min
        self.output_min = np.ones(self.num_features) * output_range[0]
        self.output_max = np.ones(self.num_features) * output_range[1]
        self.output_range = self.output_max - self.output_min
        self.scale_ = self.output_range / self.feature_range

        if len(self.feature_min) != len(self.feature_max):
            raise ValueError("feature_min and feature_max must have the same length.")

    def transform(self, x):
        """
        Transform the input data to the range [-1, 1].
        Parameters:
        - x: Array-like, shape (n_samples, n_features), input data to transform.
        Returns:
        - x_transformed: Transformed data with values in the range [-1, 1].
        """
        if x.shape[1] != len(self.feature_min):
            raise ValueError(
                f"Expected input with {len(self.feature_min)} features, but got {x.shape[1]} features."
            )

        x = np.array(x)
        x = (x - self.feature_min) * self.scale_ + self.output_min
        return x

    def inverse_transform(self, x):
        """
        Inverse transform the scaled data back to the original range.
        Parameters:
        - x_scaled: Array-like, shape (n_samples, n_features), scaled data to inverse transform.
        Returns:
        - x_original: Data transformed back to the original feature range.
        """
        if x.shape[1] != len(self.feature_min):
            raise ValueError(
                f"Expected input with {len(self.feature_min)} features, but got {x.shape[1]} features."
            )

        x = np.array(x)
        x = (x - self.output_min) / self.scale_ + self.feature_min
        return x

    def inverse_transform_ix(self, x, ix):
        """
        Inverse transform the scaled data back to the original range.
        This method applies the scaling to the indicated dimension only.

        Parameters:
        - x_scaled: Array-like, shape (n_samples, n_features), scaled data to inverse transform.

        Returns:
        - x_original: Data transformed back to the original feature range.
        """
        if ix >= len(self.feature_min):
            raise IndexError("Index out of bounds for feature dimension.")
        if len(x.shape) != 2:
            raise ValueError("Expected 2D array for x_scaled.")

        x[:, ix] = (x[:, ix] - self.output_min[ix]) / self.scale_[
            ix
        ] + self.feature_min[ix]
        return x

    def inverse_transform_mat(self, x, ix):
        """
        Inverse transform the scaled data back to the original range (of ix column).
        This works for an array of any dimensionality or shape.

        Parameters:
        - x_scaled: Array-like, shape (n_samples, n_features), scaled data to inverse transform.

        Returns:
        - x_original: Data transformed back to the original feature range.
        """
        x = (x - self.output_min[ix]) / self.scale_[ix] + self.feature_min[ix]
        return x


def run_python_script(
    script_path, function_name="main.py", verbose: bool = False, **kwargs
):
    """
    @param script_path:
    @param function_name:
    @param kwargs:
    @return:
    """

    current_directory = Path(__file__).resolve().parent
    parent_directory = current_directory.parent

    sript_path = Path(script_path)
    if sript_path.is_absolute():
        script_full_path = script_path
    else:
        script_full_path = parent_directory / script_path

    if not script_full_path.exists():
        raise FileNotFoundError(f"The Python script does not exist: {script_full_path}")

    if verbose:
        print(f"Calling Python script {script_path}")

    command = ["python3", str(script_full_path / function_name)]
    for key, value in kwargs.items():
        command.append(f"--{key}")
        if isinstance(value, dict):
            command.append(json.dumps(value))
        else:
            command.append(str(value))

    try:
        process = subprocess.Popen(command)
        return process

    except Exception as e:
        # TODO this needs to be handled or passed to the UI
        print(f"Error starting Python (process): {e}")


@DeprecationWarning
def monitor_output(output_queue):
    """Continuously reads from the output queue and prints lines as they come in."""
    while True:
        try:
            line = output_queue.get(timeout=1)  # Timeout to allow for graceful shutdown
            print(line, end="")  # Print each line from the queue
        except queue.Empty:
            continue  # No output available, keep checking


def generate_grids(
    n: int, num_points: int, upper_bound: np.ndarray, lower_bound: np.ndarray
):
    """
    Generates an evenly distributed grid across n dimensions and its midpoints.
    By default, the grid goes from 0-1

    @param n: number of dimensions
    @param num_points: number of points per dimension (same in all dimensions)
    @param upper_bound: if provided the grid is rescaled
    @param lower_bound:
    @return:
    """
    # Create an evenly spaced grid in each dimension
    points = np.linspace(0, 1, num_points)

    # Generate the full grid by creating a meshgrid and then stacking the result
    grids = np.meshgrid(*([points] * n), indexing="ij")
    grid = np.stack(grids, axis=-1).reshape(-1, n)

    # Calculate midpoints by averaging adjacent points in the grid
    midpoints = (points[:-1] + points[1:]) / 2
    midpoints_grids = np.meshgrid(*([midpoints] * n), indexing="ij")
    midpoints_grid = np.stack(midpoints_grids, axis=-1).reshape(-1, n)

    grid = lower_bound + grid * (upper_bound - lower_bound)
    midpoints_grid = lower_bound + midpoints_grid * (upper_bound - lower_bound)

    return grid, midpoints_grid


def receive_package(sock, size_lim: int = 65536):
    """
    Receives and decodes a single JSON package from a sock.

    Note: This still assumes a single recv call is sufficient to get the full message.
    For true robustness, a message framing protocol (e.g., sending size first)
    would be needed to handle the TODO.

    Args:
        sock: The active sock to read from.
        size_lim: The maximum number of bytes to receive.

    Returns:
        The decoded Python object from JSON, or None if the connection was
        closed or an error occurred during processing.
    """
    try:
        data_bytes = sock.recv(size_lim)
        # If the socket is closed by the peer, recv returns an empty bytes object.
        if not data_bytes:
            return None

        data_str = data_bytes.decode("utf-8")
        return json.loads(data_str)

    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"Error decoding package: {e}")
        return None
    except ConnectionResetError:
        print("Connection was reset by the peer.")
        return None


def fetch_data(sock: socket.socket) -> Optional[list[dict[str, Any]]]:
    """
    Retrieves a complete data payload from the socket.

    Handles both single-package and multi-package (chunked) transmissions.
    It ensures that the returned data is a list of dictionaries.

    Args:
        sock: The active client socket to receive data from.

    Returns:
        A list of dictionaries representing the complete data payload,
        or None if no data is received or the data format is invalid.
    """
    data = receive_package(sock)

    if data is None:
        return None

    # If it's a single package, return it as a list
    if isinstance(data, dict):
        return [data]

    # Check if we are dealing with chunked messages
    if isinstance(data, list):
        if not all(isinstance(item, dict) for item in data):
            print("Error: Received a list that contains non-dictionary items.")
            return None

        last_package = data[-1]
        total_packages = last_package.get("tot_pckgs")

        if isinstance(total_packages, int):
            # It's a chunked message. Receive the remaining packages.
            while len(data) < total_packages:
                next_package = receive_package(sock)
                if isinstance(next_package, dict):
                    data.append(next_package)
                else:
                    # Invalid package received mid-stream, abort.
                    print(
                        "Error: Invalid package received during chunked transmission."
                    )
                    return None

        return data  # Return the complete list of packages.

    msg = f"Warning: Received data of unexpected type: {type(data)}"
    warnings.warn(msg)
    return None


def array_to_list_of_dicts(array, column_names):
    """
    Convert a np.ndarray into a list of dictionaries
    @param array:
    @param column_names:
    @return:
    """
    if array.shape[1] != len(column_names):
        raise ValueError(
            "Number of columns in array must match the number of column names."
        )

    return [dict(zip(column_names, row)) for row in array]


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


def define_scaler_search_space(
    problem_config: dict[str, Any],
    scale_inputs: bool = True,
    scaler: str = "minmax",
    output_range: tuple[float, float] = (-1, 1),
) -> tuple[Optional[CustomBox], BaseScaler, dict[str, Any]]:
    """
    @param problem_config: Information concerning the search space as defined by the user.
                The gui boundaries should be within the problem_config range.
                If not the case, these are ignored.
    @param scale_inputs: bool, I don't see a scenario where this wouldn't be the case.
    @param scaler : string to specify the scaling type
    @param output_range : tuple indicating min and max value in the output
    @return: tuple : CustomBox, BaseScaler, feature_info
    """
    # TODO extend to handle categorical and boolean features (i.e. they have to bypass the feature normalization
    if scaler != "minmax":
        raise NotImplementedError(f"Scaler '{scaler}' has not been implemented.")

    feature_info = {"fixed": {}, "variable": {}, "categorical": {}}

    lower_bound = []
    upper_bound = []

    for name, config in problem_config.items():
        _parse_parameter(name, config, feature_info, lower_bound, upper_bound)
    print("Feature info:", feature_info)

    if not lower_bound and not upper_bound:
        scaler_obj = IdentityScaler()
        raise NotImplementedError("No implementation for only categorical problems ")
        # return None, scaler_obj, feature_info

    if scale_inputs:
        num_features = len(lower_bound)
        search_space = CustomBox(
            lower=num_features * [output_range[0]],
            upper=num_features * [output_range[1]],
        )
        scaler_obj = CustomMinMaxScaler(
            feature_min=lower_bound, feature_max=upper_bound, output_range=output_range
        )
    else:
        search_space = CustomBox(lower=lower_bound, upper=upper_bound)
        scaler_obj = IdentityScaler()

    return search_space, scaler_obj, feature_info

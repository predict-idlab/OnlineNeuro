# /online_neuro/utils.py
import numpy as np
import matlab.engine
import json
from typing import List, Optional, Union
import subprocess
import queue
from trieste.space import Box
from pathlib import Path
import time
import select
import warnings


class CustomBox(Box):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample_method(self, num_samples: int,
                      seed: Optional[int] = None,
                      skip: Optional[int] = None,
                      max_tries: int = 100,
                      sampling_method: str = 'sobol'):
        if sampling_method == 'random':
            return self.sample(num_samples, seed)
        elif sampling_method == 'halton':
            return self.sample_halton(num_samples, seed)
        elif sampling_method == 'sobol':
            return self.sample_sobol(num_samples, skip)
        elif sampling_method == 'random_feasible':
            return self.sample_feasible(num_samples, seed, max_tries)
        elif sampling_method == 'halton_feasible':
            return self.sample_feasible(num_samples, seed, max_tries)
        elif sampling_method == 'sobol_feasible':
            return self.sample_feasible(num_samples, skip, max_tries)
        else:
            raise ValueError(f'Unsupported sampling method: {sampling_method}')


class SearchSpacePipeline:
    def __init__(self, search_space,
                 mapping: dict, feature_names: Union[List, np.ndarray],
                 scaler=None):
        """
        @param search_space:
        @param mapping:
        @param feature_names:
        @param scaler:
        """
        self.search_space = search_space
        self.mapping = mapping
        self.feature_names = feature_names
        self.scaler = scaler

    def sample(self, num_samples, sample_method='sobol'):
        if sample_method == 'sobol':
            samples = self.search_space.sample_sobol(num_samples)
        else:
            raise NotImplementedError(f'Sampling method {sample_method} not implemented (yet).')

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
            sample_dict = {k: None for k in self.feature_names}
            for k in self.feature_names:
                sample_dict[k] = x[sample_idx, self.mapping[k]]

            as_list.append(sample_dict)

        if with_array:
            return as_list, x
        else:
            return as_list

    def feature_names(self):
        return self.feature_names

    def feature_names_ixs(self):
        feature_names_ixs = []
        for fn in self.feature_names:
            feat_ixs_list = [f'{fn}_{ix}' for ix in range(len(self.mapping[fn]))]
            feature_names_ixs.extend(feat_ixs_list)

        return feature_names_ixs


class CustomMinMaxScaler:
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
            raise ValueError('feature_min and feature_max must have the same length.')

    def transform(self, x):
        """
        Transform the input data to the range [-1, 1].
        Parameters:
        - x: Array-like, shape (n_samples, n_features), input data to transform.
        Returns:
        - x_transformed: Transformed data with values in the range [-1, 1].
        """
        if x.shape[1] != len(self.feature_min):
            raise ValueError(f'Expected input with {len(self.feature_min)} features, but got {x.shape[1]} features.')

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
            raise ValueError(f'Expected input with {len(self.feature_min)} features, but got {x.shape[1]} features.')

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
            raise IndexError('Index out of bounds for feature dimension.')
        if len(x.shape) != 2:
            raise ValueError('Expected 2D array for x_scaled.')

        x[:, ix] = (x[:, ix] - self.output_min[ix]) / self.scale_[ix] + self.feature_min[ix]
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


def run_matlab(matlab_script_path, matlab_function_name='main', **kwargs):
    """
    Launch a matlab process
    @param matlab_script_path:
    @param matlab_function_name:
    @return:

    """
    current_directory = Path(__file__).resolve().parent
    parent_directory = current_directory.parent

    if Path(matlab_script_path).is_absolute():
        matlab_script_full_path = Path(matlab_script_path)
    else:
        # If it's a relative path, join with current_directory
        # Here we assume that problems will be stored at ./simulators/matlab/problems
        matlab_script_full_path = parent_directory / matlab_script_path

    if not matlab_script_full_path.exists():
        raise FileNotFoundError(f'The MATLAB folder does not exist: {matlab_script_full_path}')

    matlab_script_full_path = str(matlab_script_full_path)
    # Start MATLAB engine
    print('Calling Matlab from Python engine')

    eng = matlab.engine.start_matlab()
    eng.cd(matlab_script_full_path)
    eng.addpath(str(parent_directory), nargout=0)
    # Call MATLAB function main
    matlab_args = json.dumps(kwargs)

    try:
        eng.feval(matlab_function_name, matlab_args, nargout=0)
        eng.quit()

    except Exception as e:
        print(f'Error starting MATLAB: {e}')


def run_python_script(script_path, function_name='main.py', **kwargs):
    """
    @param script_path:
    @param function_name:
    @param kwargs:
    @return:
    """
    print('Calling Python script from within Python')
    current_directory = Path(__file__).resolve().parent
    parent_directory = current_directory.parent

    sript_path = Path(script_path)
    if sript_path.is_absolute():
        script_full_path = script_path
    else:
        script_full_path = parent_directory / script_path

    if not script_full_path.exists():
        raise FileNotFoundError(f'The Python script does not exist: {script_full_path}')

    command = ['python3', str(script_full_path / function_name)]
    for key, value in kwargs.items():
        command.append(f'--{key}')
        if isinstance(value, dict):
            command.append(json.dumps(value))
        else:
            command.append(str(value))

    try:
        process = subprocess.Popen(command)
        return process

    except Exception as e:
        print(f'Error starting Python (process): {e}')


def monitor_output(output_queue):
    """Continuously reads from the output queue and prints lines as they come in."""
    while True:
        try:
            line = output_queue.get(timeout=1)  # Timeout to allow for graceful shutdown
            print(line, end='')  # Print each line from the queue
        except queue.Empty:
            continue  # No output available, keep checking


def generate_grids(n, num_points, upper_bound=None, lower_bound=None):
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
    grids = np.meshgrid(*([points] * n), indexing='ij')
    grid = np.stack(grids, axis=-1).reshape(-1, n)

    # Calculate midpoints by averaging adjacent points in the grid
    midpoints = (points[:-1] + points[1:]) / 2
    midpoints_grids = np.meshgrid(*([midpoints] * n), indexing='ij')
    midpoints_grid = np.stack(midpoints_grids, axis=-1).reshape(-1, n)

    if upper_bound and lower_bound:
        grid = lower_bound + grid * (upper_bound - lower_bound)
        midpoints_grid = lower_bound + midpoints_grid * (upper_bound - lower_bound)
    else:
        msg = 'Both upper_bound and lower_bound need to be defined. Scaling was ignored'
        warnings.warn(msg)
    return grid, midpoints_grid


# def write_data(client_socket, data):
#     json_data = json.dumps(data, ensure_ascii=False).encode('utf-8')
#     try:
#         client_socket.sendall(json_data)
#         return {'Message': 'Data sent to server'}
#     except Exception as e:
#         msg = e
#         return {'Message': f'Something went wrong {msg}'}


def receive_data(socket, buffer_size=65536, timeout=300):
    """
    Receives JSON data over a socket connection.
    @param socket: The socket to receive data from.
    @param buffer_size: The size of the buffer for each read operation.
    @param timeout: Timeout in seconds for receiving data.
    @return: Decoded Python object or raises an error.
    """
    buffer = b''
    end_time = time.time() + timeout

    while time.time() < end_time:
        ready_to_read, _, _ = select.select([socket], [], [], 1)
        if ready_to_read:
            part = socket.recv(buffer_size)
            if part:
                buffer += part
                print(len(buffer))
                print(buffer)
            # Check for complete JSON message(s) in the buffer
            if b'\\n' in buffer:
                messages = buffer.split(b'\\n')
                for message in messages[:-1]:  # Process complete messages
                    if message.strip():  # Ignore empty messages
                        try:
                            return json.loads(message.decode('utf-8').strip())
                        except json.JSONDecodeError as e:
                            print(f'Error decoding JSON: {e}, Message: {message}')

                # Retain incomplete part in the buffer
                buffer = messages[-1]

    raise TimeoutError('Timeout: No complete JSON message received.')


def receive_data(socket, size_lim=65536):
    """ Placeholder function, assumes data is received fully. """
    data = socket.recv(size_lim)  # Read from socket
    return json.loads(data.decode())  # Decode JSON properly


def fetch_data(socket):
    data = receive_data(socket)

    # If it's a single package, return it as a list
    if isinstance(data, dict):
        return [data]

    # Check if we are dealing with chunked messages
    if isinstance(data, list) and isinstance(data[-1], dict):
        if 'tot_pckgs' in data[-1] and 'current_pckg' in data[-1]:
            total_packages = data[-1]['tot_pckgs']
            received_packages = len(data)

            # Keep receiving if we haven't gotten all packages
            while received_packages < total_packages:
                msg = receive_data(socket)
                data.append(msg)
                received_packages += 1

            return data  # Return full data
        else:
            return data  # Return as is

    # If data format is unexpected, wrap it in a list
    return [data]

def array_to_list_of_dicts(array, column_names):
    """
    Convert a np.ndarray into a list of dictionaries
    @param array:
    @param column_names:
    @return:
    """
    if array.shape[1] != len(column_names):
        raise ValueError('Number of columns in array must match the number of column names.')

    return [dict(zip(column_names, row)) for row in array]


def define_scaler_search_space(problem_config,
                               scale_inputs: bool = True,
                               scaler: str = 'minmax',
                               output_range: tuple = (-1, 1)):
    """
    @param problem_config: Information concerning the search space as defined by the user.
                The gui boundaries should be within the problem_config range.
                If not the case, these are ignored.
    @param scale_inputs: bool, I don't see a scenario where this wouldn't be the case.
    @param scaler : string to specify the scaling type
    @param output_range : tuple indicating min and max value in the output
    @return: instance : SearchSpacePipeline
    """
    # TODO extend to handle categorical and boolean features (i.e. they have to bypass the feature normalization
    fixed_features = {}
    variable_features = {}

    lower_bound = []
    upper_bound = []

    for key, value in problem_config.items():
        if isinstance(value, list):
            for ix in range(len(value)):
                element = value[ix]
                if isinstance(element, dict):
                    for k1, v1 in element.items():
                        feat_name = f'{key}_{ix + 1}_{k1}'
                        if 'min_value' in v1 and 'max_value' in v1:
                            if v1['min_value'] == v1['max_value']:
                                fixed_features[feat_name] = v1['min_value']
                            else:
                                variable_features[feat_name] = 0
                                lower_bound.append(v1['min_value'])
                                upper_bound.append(v1['max_value'])
                        elif 'value' in v1:
                            fixed_features[feat_name] = v1['value']
                        else:
                            msg = f"Feature '{key}' does not have a valid formatting, skipping"
                            warnings.warn(msg)
                else:
                    # Contains a fixed list, no need to iterate through each element
                    fixed_features[key] = value
                    continue
        elif isinstance(value, dict):
            if 'min_value' in problem_config[key] and 'max_value' in problem_config[key]:
                if isinstance(problem_config[key]['value'], list):
                    for ix in range(len(problem_config[key]['value'])):
                        if len(problem_config[key]['value']) == 1:
                            feat_name = key
                        else:
                            feat_name = f'{key}_{ix + 1}'

                        if problem_config[key]['min_value'][ix] == problem_config[key]['max_value'][ix]:
                            fixed_features[feat_name] = value
                        else:
                            variable_features[feat_name] = 0
                            lower_bound.append(problem_config[key]['min_value'][ix])
                            upper_bound.append(problem_config[key]['max_value'][ix])
                else:
                    feat_name = key
                    if problem_config[key]['min_value'] == problem_config[key]['max_value']:
                        fixed_features[feat_name] = value
                    else:
                        variable_features[feat_name] = 0
                        lower_bound.append(problem_config[key]['min_value'])
                        upper_bound.append(problem_config[key]['max_value'])
            elif 'value' in problem_config[key]:
                fixed_features[key] = value
            else:
                msg = f"Feature '{key}' does not have a valid formatting, skipping"
                warnings.warn(msg)

        elif isinstance(value, (list, float, int, str)):
            fixed_features[key] = value

        else:
            ty = type(value)
            msg = f"Feature '{key}' does not have a valid type '{ty}', skipping"
            warnings.warn(msg)

    if scale_inputs:
        num_features = len(lower_bound)

        # TODO, implement other types of normalization
        if scaler == 'minmax':
            lb = num_features * [output_range[0]]
            ub = num_features * [output_range[1]]

            search_space = CustomBox(lower=lb,
                                     upper=ub)

            scaler = CustomMinMaxScaler(feature_min=lower_bound,
                                        feature_max=upper_bound,
                                        output_range=output_range)
        else:
            raise NotImplementedError(f'{scaler} has not been implemented')
    else:
        search_space = CustomBox(lower=lower_bound,
                                 upper=upper_bound)
        scaler = None

    feat_dict = {'fixed_features': fixed_features,
                 'variable_features': variable_features}
    return search_space, scaler, feat_dict

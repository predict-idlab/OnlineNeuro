import os
import numpy as np
import matlab.engine
import json
from trieste.space import SearchSpaceType
from typing import List, Optional, Sequence, Callable, Hashable, Tuple, TypeVar, Union

class SearchSpacePipeline:
    def __init__(self, search_space: SearchSpaceType, mapping: dict, feature_names: Union[List, np.ndarray], scaler=None):
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
            raise NotImplementedError(f"Sampling method {sample_method} not implemented (yet).")

        samples = self.inverse_transform(samples.numpy())
        return samples

    def inverse_transform(self, x, with_array=True):
        """
        @param x:
        @param with_array:
        @return:
        """
        if self.scaler:
            scaled_samples = self.scaler.inverse_transform(x)
        reconstructed_list = []

        for sample_idx in range(x.shape[0]):
            sample_dict = {k: None for k in self.feature_names}
            for k in self.feature_names:
                sample_dict[k] = x[sample_idx, self.mapping[k]]

            reconstructed_list.append(sample_dict)

        if with_array:
            return reconstructed_list, scaled_samples
        else:
            return reconstructed_list

    def feature_names(self):
        return self.feature_names()

    def feature_names_ixs(self):
        feature_names_ixs = []
        for fn in self.feature_names:
            feat_ixs_list = [f"{fn}_{ix}" for ix in range(len(self.mapping[fn]))]
            feature_names_ixs.extend(feat_ixs_list)

        return feature_names_ixs


class customMinMaxScaler:
    def __init__(self, feature_min, feature_max, output_range=(0, 1)):
        """
        Initialize the scaler with min and max bounds for each feature.

        Parameters:
        - feature_min: Array-like of minimum values for each feature.
        - feature_max: Array-like of maximum values for each feature.
        - output_range: Range to scale outputs.
        @note This function does not raise warnings if future samples are above or bellow the expected min, maxs

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

        self.feature_min = np.array(feature_min)
        self.feature_max = np.array(feature_max)
        self.feature_range = self.feature_max - self.feature_min
        self.output_min, self.output_max = output_range
        self.output_range = self.output_max - self.output_min
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
            raise ValueError(f"Expected input with {len(self.feature_min)} features, but got {x.shape[1]} features.")

        x = np.array(x)
        x_scaled = (x - self.feature_min) / self.feature_range
        x_transformed = x_scaled * (self.output_max - self.output_min) + self.output_min
        return x_transformed

    def inverse_transform(self, x_scaled):
        """
        Inverse transform the scaled data back to the original range.

        Parameters:
        - x_scaled: Array-like, shape (n_samples, n_features), scaled data to inverse transform.

        Returns:
        - x_original: Data transformed back to the original feature range.
        """
        if x_scaled.shape[1] != len(self.feature_min):
            raise ValueError(f"Expected input with {len(self.feature_min)} features, but got {x.shape[1]} features.")

        x_scaled = np.array(x_scaled)
        x_original = (x_scaled - self.output_min) / (self.output_max - self.output_min)
        x_original = x_original * self.feature_range + self.feature_min
        return x_original

    def inverse_transform_ix(self, x_scaled, ix):
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
        if len(x_scaled.shape) != 2:
            raise ValueError("Expected 2D array for x_scaled.")

        x_scaled = np.array(x_scaled)
        x_original = (x_scaled[:, ix] - self.output_min) / self.output_range  # scale to [0, 1]
        x_original = x_original * self.feature_range[ix] + self.feature_min[ix]  # scale to original feature range
        return x_original

    def inverse_transform_mat(self, x_scaled, ix):
        """
        Inverse transform the scaled data back to the original range (of ix column).
        This works for an array of any dimensionality or shape.

        Parameters:
        - x_scaled: Array-like, shape (n_samples, n_features), scaled data to inverse transform.

        Returns:
        - x_original: Data transformed back to the original feature range.
        """
        x_scaled = np.array(x_scaled)
        x_original = (x_scaled - self.output_min) / self.output_range  # scale to [0, 1]
        x_original = x_original * self.feature_range[ix] + self.feature_min[ix]  # scale to original feature range
        return x_original


def run_matlab(matlab_script_path, matlab_function_name='main', **kwargs):
    """
    Launch a matlab process
    @param matlab_script_path:
    @param matlab_function_name:
    @return:

    """
    current_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.dirname(current_directory)

    if os.path.isabs(matlab_script_path):
        matlab_script_full_path = matlab_script_path
    else:
        # If it's a relative path, join with current_directory
        # Here we assume that problems will be stored at ./simulators/matlab/problems
        matlab_script_full_path = os.path.join(parent_directory, matlab_script_path)

    if not os.path.exists(matlab_script_full_path):
        raise FileNotFoundError(f"The MATLAB folder does not exist: {matlab_script_full_path}")

    # Start MATLAB engine
    print("Calling Matlab from Python engine")

    eng = matlab.engine.start_matlab()
    eng.cd(matlab_script_full_path)
    eng.addpath(parent_directory, nargout=0)
    # Call MATLAB function main
    matlab_args = json.dumps(kwargs)

    print(matlab_args)
    try:
        eng.feval(matlab_function_name, matlab_args, nargout=0)
        eng.quit()

    except Exception as e:
        print(f"Error starting MATLAB: {e}")


def generate_grids(n, num_points):
    """
    Generates an evenly distributed grid across n dimensions and its midpoints.

    Parameters:
        n (int): Number of dimensions.
        num_points (int): Number of points per dimension.

    Returns:
        tuple: A tuple containing:
            - grid (np.ndarray): The evenly distributed grid.
            - midpoints_grid (np.ndarray): The midpoints of the grid.

    # Example usage
    n = 3  # Number of dimensions
    num_points = 4  # Number of points per dimension
    grid, midpoints_grid = generate_grids(n, num_points)

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

    return grid, midpoints_grid

def fetch_data(client_socket):
    received_data = client_socket.recv(1024).decode()
    received_data = json.loads(received_data)
    if 'tot_pckgs' in received_data:
        tot_packages = received_data['tot_pckgs']
        all_data = []
        all_data.append(received_data['data'])
        for _ in range(tot_packages - 1):
            msg = client_socket.recv(1024).decode()
            msg = json.loads(msg)
            all_data.append(msg['data'])

        #Flatten
        all_data = [x for xs in all_data for x in xs]
        return all_data
    else:
        return received_data
import os
import numpy as np
import matlab.engine
import json
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
    def transform(self, X):
        """
        Transform the input data to the range [-1, 1].

        Parameters:
        - X: Array-like, shape (n_samples, n_features), input data to transform.

        Returns:
        - X_transformed: Transformed data with values in the range [-1, 1].
        """
        X = np.array(X)
        X_transformed = (X - self.feature_min) / self.feature_range
        X_transformed = X_transformed * self.output_range + self.output_min
        return X_transformed

    def inverse_transform(self, X_scaled):
        """
        Inverse transform the scaled data back to the original range.

        Parameters:
        - X_scaled: Array-like, shape (n_samples, n_features), scaled data to inverse transform.

        Returns:
        - X_original: Data transformed back to the original feature range.
        """
        X_scaled = np.array(X_scaled)
        X_original = (X_scaled - self.output_min) / self.output_range  # scale to [0, 1]
        X_original = X_original * self.feature_range + self.feature_min  # scale to original feature range
        return X_original

    def inverse_transform_ix(self, X_scaled, ix):
        """
        Inverse transform the scaled data back to the original range.
        This method applies the scaling to the indicated dimension only.

        Parameters:
        - X_scaled: Array-like, shape (n_samples, n_features), scaled data to inverse transform.

        Returns:
        - X_original: Data transformed back to the original feature range.
        """
        X_scaled = np.array(X_scaled)
        X_original = (X_scaled[:, ix] - self.output_min) / self.output_range  # scale to [0, 1]
        X_original = X_original * self.feature_range[ix] + self.feature_min[ix]  # scale to original feature range
        return X_original

    def inverse_transform_mat(self, X_scaled, ix):
        """
        Inverse transform the scaled data back to the original range (of ix column).
        This works for an array of any dimensionality or shape.

        Parameters:
        - X_scaled: Array-like, shape (n_samples, n_features), scaled data to inverse transform.

        Returns:
        - X_original: Data transformed back to the original feature range.
        """
        X_scaled = np.array(X_scaled)
        X_original = (X_scaled - self.output_min) / self.output_range  # scale to [0, 1]
        X_original = X_original * self.feature_range[ix] + self.feature_min[ix]  # scale to original feature range
        return X_original

def run_matlab_main(matlab_script_path=None, matlab_function_name='main', **kwargs):
    """
    Launch a matlab process
    @param kwargs:
    @return:
    # TODO If needed, modify this function so that arguments are passed to the matlab engine
    # TODO modify function so that path is parameter (instead of file location)
    """
    # Start MATLAB engine
    print("Launching Matlab from Python side")
    eng = matlab.engine.start_matlab()

    # Change directory to current directory
    if matlab_script_path is None:
        current_directory = os.path.dirname(os.path.abspath(__file__))
        eng.cd(current_directory)  # Change path to your directory where main.m resides
    else:
        eng.cd(matlab_script_path)

    # Call MATLAB function main
    matlab_args = []
    if 'matlab_initiate' in kwargs:
        initiate_bool = bool(kwargs['matlab_initiate'])
        matlab_args.append(initiate_bool)

    for key, value in kwargs.items():
        if key != 'matlab_initiate':
            matlab_args.append(value)

    #Converting to Matlab format (if arrays)
    matlab_args = [eng.convert_to_mlarray(arg) if isinstance(arg, list) else arg for arg in matlab_args]
    eng.feval(matlab_function_name, *matlab_args, nargout=0)

    # Stop MATLAB engine
    eng.quit()

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
        rest_of_msg = []
        for _ in range(received_data['tot_pckgs'] - 1):
            msg = client_socket.recv(1024).decode()
            msg = json.loads(msg)
            rest_of_msg.append(msg)

        for msg in rest_of_msg:
            for k, v in msg.items():
                if k != 'tot_pckgs':
                    received_data[k] += v
    return received_data
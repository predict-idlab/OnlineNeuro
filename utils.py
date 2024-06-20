import numpy as np
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
        Inverse transform the scaled data back to the original range.

        Parameters:
        - X_scaled: Array-like, shape (n_samples, n_features), scaled data to inverse transform.

        Returns:
        - X_original: Data transformed back to the original feature range.
        """
        X_scaled = np.array(X_scaled)
        X_original = (X_scaled - self.output_min) / self.output_range  # scale to [0, 1]
        X_original = X_original * self.feature_range[ix] + self.feature_min[ix]  # scale to original feature range
        return X_original

# feature_min = [0, 10, -1]  # Min bounds for each feature
# feature_max = [100, 50, 1]  # Max bounds for each feature
#
# scaler = customMinMaxScaler(feature_min, feature_max, output_range=(-1, 1))
#
# X = np.array([[0, 10, -1], [50, 30, 0], [75, 40, -0.5], [100, 50, 0.9999]])
# X_scaled = scaler.transform(X)
# print("Scaled Data:\n", X_scaled)
#
# X_original = scaler.inverse_transform(X_scaled)
# print("Original Data:\n", X_original)

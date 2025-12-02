# /online_neuro/utils/scalers.py
from typing import Any

import numpy as np

from online_neuro.utils.config_parser import _parse_parameter
from online_neuro.utils.custom_box import CustomBox

ArrayLike = np.ndarray | list


class BaseScaler:
    """
    Abstract Base Class defining the required interface for a search space scaler.

    This class serves as a template for scalers, ensuring that any subclass
    will have the required transformation methods. It is designed to satisfy
    static analysis tools like Pylance by defining a consistent API.

    Subclasses must implement the 'transform' and 'inverse_transform' methods,
    and raise a NotImplementedError for any unimplemented methods.
    """

    def __init__(
        self,
        feature_min: ArrayLike,
        feature_max: ArrayLike,
        output_range: tuple[float, float],
    ):
        """
        Initialize the scaler with min and max bounds for each feature.
        This base implementation does not store or process these values.

        Parameters
        ----------
        feature_min : ArrayLike
            The minimum bound for each feature in the original space.
            Expected shape is (D,), where D is the number of features.
        feature_max : ArrayLike
            The maximum bound for each feature in the original space.
            Expected shape is (D,).
        output_range : tuple of (float, float), optional
            The target range for the transformed data (e.g., (0, 1) for
            normalization).
        """
        pass

    def transform(self, x: ArrayLike) -> np.ndarray:
        """
        Transform the input data to the specified output range.
        This method must be overridden by a subclass.
        """
        raise NotImplementedError("Subclasses must implement the 'transform' method.")

    def inverse_transform(self, x: ArrayLike) -> np.ndarray:
        """
        Inverse transform the scaled data back to the original range.
        This method must be overridden by a subclass.
        """
        raise NotImplementedError(
            "Subclasses must implement the 'inverse_transform' method."
        )

    def inverse_transform_ix(self, x: ArrayLike, ix: int) -> np.ndarray:
        """
        Inverse transform a single feature of the scaled data.
        This method must be overridden by a subclass.
        """
        raise NotImplementedError(
            "Subclasses must implement the 'inverse_transform_ix' method."
        )

    def inverse_transform_mat(self, x: ArrayLike, ix: int) -> np.ndarray:
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
        """
        Initializes the IdentityScaler.

        It accepts all arguments defined by BaseScaler's signature
        (`feature_min`, `feature_max`, `output_range`) but ignores them,
        ensuring compatibility with functions that expect a scaler object
        with defined bounds.
        """
        pass

    def transform(self, x: ArrayLike) -> np.ndarray:
        """Performs the forward identity transformation."""
        if isinstance(x, list):
            x = np.array(x)
        return x

    def inverse_transform(self, x: ArrayLike) -> np.ndarray:
        """Performs the inverse identity transformation."""
        if isinstance(x, list):
            x = np.array(x)
        return x

    def inverse_transform_ix(self, x: ArrayLike, ix: int) -> np.ndarray:
        """Performs the inverse identity transformation for a single feature."""
        if isinstance(x, list):
            x = np.array(x)
        return x

    def inverse_transform_mat(self, x: ArrayLike, ix: int) -> np.ndarray:
        """Performs the inverse identity transformation for a matrix using a single feature's scaling parameters."""
        if isinstance(x, list):
            x = np.array(x)
        return x


class CustomMinMaxScaler(BaseScaler):
    """
    A Min-Max scaler implementation that scales data based on predefined
    feature minimum and maximum bounds to a specified output range.
    """

    def __init__(
        self,
        feature_min: ArrayLike,
        feature_max: ArrayLike,
        output_range: tuple[float, float] = (0.0, 1.0),
    ):
        """
        Initialize the scaler by calculating the transformation parameters.

        The transformation operates based on the formula:
        X_scaled = (X - X_min) * scale_ + output_min

        Parameters
        ----------
        feature_min : ArrayLike
            Array-like (shape D,) of minimum values for each feature in the
            original space.
        feature_max : ArrayLike
            Array-like (shape D,) of maximum values for each feature in the
            original space.
        output_range : tuple of (float, float), optional
            The target range (min, max) for the scaled data. Default is (0.0, 1.0).

        Raises
        ------
        ValueError
            If `feature_min` and `feature_max` do not have the same length (number of features).

        Notes
        -----
        This implementation does not raise warnings if input samples during
        `transform` fall outside the defined `feature_min` or `feature_max` bounds.
        It silently performs a linear extrapolation based on the defined scaling.

        Examples
        --------
        >>> feature_min = [0, 10, -1]  # Min bounds for each feature
        >>> feature_max = [100, 50, 1]  # Max bounds for each feature
        >>> scaler = CustomMinMaxScaler(feature_min, feature_max, output_range=(-1, 1))

        >>> X = np.array([[0, 10, -1], [50, 30, 0], [100, 50, 0.9999]])
        >>> X_scaled = scaler.transform(X)
        >>> X_scaled.round(2)
        array([[-1.  , -1.  , -1.  ],
               [ 0.  ,  0.  ,  0.  ],
               [ 1.  ,  1.  ,  1.  ]])

        >>> X_original = scaler.inverse_transform(X_scaled)
        >>> X_original.round(2)
        array([[  0.  ,  10.  ,  -1.  ],
               [ 50.  ,  30.  ,   0.  ],
               [100.  ,  50.  ,   1.  ]])
        """

        self.num_features = len(feature_min)
        self.feature_min = np.asarray(feature_min, dtype=float)
        self.feature_max = np.asarray(feature_max, dtype=float)

        if len(self.feature_min) != len(self.feature_max):
            raise ValueError("feature_min and feature_max must have the same length.")

        self.feature_range = self.feature_max - self.feature_min
        locs = np.where(self.feature_range == 0)[0]
        if np.any(locs):
            raise ValueError(
                f"Feature(s) at index {locs.tolist()} have zero range (min equals max)."
            )

        self.output_min = np.ones(self.num_features) * output_range[0]
        self.output_max = np.ones(self.num_features) * output_range[1]
        self.output_range = self.output_max - self.output_min
        self.scale_ = self.output_range / self.feature_range

    def transform(self, x: ArrayLike) -> np.ndarray:
        """
        Transforms the input data from the original space to the defined output range.

        Parameters
        ----------
        x : ArrayLike
            Input data to transform, shape (N, D).

        Returns
        -------
        np.ndarray
            The transformed data, shape (N, D), scaled to the `output_range`.

        Raises
        ------
        ValueError
            If the number of features in `x` does not match the dimensions
            the scaler was initialized with.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[np.newaxis, :]

        if x.shape[-1] != self.num_features:
            raise ValueError(
                f"Expected input with {self.num_features} features, but got {x.shape[-1]} features."
            )
        x = (x - self.feature_min) * self.scale_ + self.output_min
        return x

    def inverse_transform(self, x: ArrayLike) -> np.ndarray:
        """
        Inverse transforms the scaled data back to the original feature range.

        Parameters
        ----------
        x : ArrayLike
            Scaled input data, shape (N, D).

        Returns
        -------
        np.ndarray
            Data transformed back to the original feature range.

        Raises
        ------
        ValueError
            If the number of features in `x` does not match the dimensions
            the scaler was initialized with.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[np.newaxis, :]

        if x.shape[-1] != self.num_features:
            raise ValueError(
                f"Expected input with {self.num_features} features, but got {x.shape[-1]} features."
            )

        # Formula: x_original = (x_scaled - output_min) / scale_ + feature_min
        x = (x - self.output_min) / self.scale_ + self.feature_min
        return x

    def inverse_transform_ix(self, x: ArrayLike, ix: int) -> np.ndarray:
        """
        Inverse transforms only the specified feature (column) of the 2D array `x`.

        The transformation is applied in-place on the specified column.

        Parameters
        ----------
        x : np.ndarray
            The 2D scaled data array, shape (N, D). The array is modified in-place.
        ix : int
            The index of the feature (column) to inverse transform.

        Returns
        -------
        np.ndarray
            The input array `x` with the specified column inverse-transformed.

        Raises
        ------
        IndexError
            If `ix` is out of bounds for the number of features.
        ValueError
            If the input `x` is not a 2D array.
        """
        x = np.asarray(x, dtype=float)
        if ix >= self.num_features:
            raise IndexError("Index out of bounds for feature dimension.")
        if x.ndim != 2:
            raise ValueError("Expected 2D array for x.")

        # Apply inverse formula only to the specified column
        x[:, ix] = (x[:, ix] - self.output_min[ix]) / self.scale_[
            ix
        ] + self.feature_min[ix]
        return x

    def inverse_transform_mat(self, x: ArrayLike, ix: int) -> np.ndarray:
        """
        Inverse transforms an array `x` of any dimensionality using the scaling
        parameters of a single feature specified by `ix`.

        This is useful for transforming auxiliary data (like covariance matrices)
        where all dimensions should be scaled identically according to one feature's bounds.

        Parameters
        ----------
        x : ArrayLike
            The scaled input array (can be N-dimensional).
        ix : int
            The index of the feature whose scaling parameters are to be used.

        Returns
        -------
        np.ndarray
            The inverse-transformed array.

        Raises
        ------
        IndexError
            If `ix` is out of bounds for the number of features.
        """
        x_arr = np.asarray(x, dtype=float)

        if ix >= self.num_features:
            raise IndexError("Index out of bounds for feature dimension.")

        # Apply inverse formula uniformly using only the index 'ix' parameters
        x_arr = (x_arr - self.output_min[ix]) / self.scale_[ix] + self.feature_min[ix]

        return x_arr


def define_scaler_search_space(
    problem_config: dict[str, Any],
    scale_inputs: bool = True,
    scaler: str = "minmax",
    output_range: tuple[float, float] = (-1, 1),
    verbose: bool = False,
) -> tuple[CustomBox, BaseScaler, dict[str, Any]]:
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

    if verbose:
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

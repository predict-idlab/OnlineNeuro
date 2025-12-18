# online_neuro/custom_architectures.py
from enum import Enum, auto
from typing import Any, Iterable, List, Union

import dill
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions.distribution import Distribution
from trieste.data import Dataset
from trieste.models.keras.architectures import (
    GaussianNetwork,
    MultivariateNormalTriL,
)
from trieste.models.keras.interface import KerasPredictor
from trieste.types import TensorType


class TaskType(Enum):
    REGRESSION = auto()
    BINARY_CLASSIFICATION = auto()
    MULTIOUTPUT_REGRESSION = auto()
    MULTICLASS_CLASSIFICATION = auto()


TASK_ALIASES = {
    "regression": TaskType.REGRESSION,
    "reg": TaskType.REGRESSION,
    "mse": TaskType.REGRESSION,
    "binary": TaskType.BINARY_CLASSIFICATION,
    "binary_classification": TaskType.BINARY_CLASSIFICATION,
    "multi_regression": TaskType.MULTIOUTPUT_REGRESSION,
    "multioutput_regression": TaskType.MULTIOUTPUT_REGRESSION,
    "multiclass_classification": TaskType.MULTICLASS_CLASSIFICATION,
    "multiclass": TaskType.MULTICLASS_CLASSIFICATION,
    "categorical": TaskType.MULTICLASS_CLASSIFICATION,
    "classification": TaskType.MULTICLASS_CLASSIFICATION,
}


def to_list(val: Union[Any, Iterable[Any]]) -> List[Any]:
    """
    Ensure that a value is returned as a list.

    Scalars are wrapped into a single-element list, while iterable inputs
    (except strings and bytes) are converted to lists.

    Parameters
    ----------
    val : Any or Iterable[Any]
        Input value to normalize.

    Returns
    -------
    list[Any]
        A list representation of the input.
    """
    if isinstance(val, (str, bytes)):  # avoid iterating over strings
        return [val]
    try:
        iter(val)
        return list(val)
    except TypeError:
        return [val]


class MonteCarloDropoutNetwork:
    """
    Fully connected neural network with Monte Carlo Dropout for uncertainty estimation.

    This class defines a feedforward neural network where dropout layers are kept
    active at inference time to approximate Bayesian uncertainty.
    """

    def __init__(
        self,
        input_tensor_spec: tf.TensorSpec,
        output_tensor_spec: tf.TensorSpec,
        hidden_layers: int | tuple[int, ...] | list[int] = 32,
        dropout_rates: float | tuple[float, ...] | list[float] = 0.0,
        activations: (
            str | tf.keras.layers.Activation | list[str | tf.keras.layers.Activation]
        ) = "relu",
        task_type: TaskType | str = TaskType.REGRESSION,
        network_name: str = "",
    ) -> None:
        """
        Initialize a Monte Carlo Dropout neural network definition.

        Parameters
        ----------
        input_tensor_spec : tf.TensorSpec
            Specification of the input tensor.
        output_tensor_spec : tf.TensorSpec
            Specification of the output tensor.
        hidden_layers : int or sequence of int, optional
            Number of units in each hidden layer.
        dropout_rates : float or sequence of float, optional
            Dropout rate for each hidden layer.
        activations : str, Activation, or sequence, optional
            Activation function(s) for each hidden layer.
        task_type : str, TaskType
            Type of task performed. Determines the output layer.
        network_name : str, optional
            Prefix used to name input and output layers.
        """
        hidden_layers = to_list(hidden_layers)
        dropout_rates = to_list(dropout_rates)
        activations = to_list(activations)

        lengths = {len(hidden_layers), len(dropout_rates), len(activations)}

        if len(lengths) != 1:
            raise ValueError(
                f"Lengths of hidden_layers ({len(hidden_layers)}), dropout_rates ({len(dropout_rates)}), and activations ({len(activations)}) must match."
            )

        self.input_tensor_spec = input_tensor_spec
        self.output_tensor_spec = output_tensor_spec
        self.hidden_layers = hidden_layers
        self.dropout_rates = dropout_rates
        self.activations = activations
        self.network_name = network_name
        self.task_type = task_type

        # Model get's instantiated
        self._model: tf.keras.Model | None = None

    @property
    def input_layer_name(self) -> str:
        """Return the name of the input layer."""
        return f"{self.network_name}input"

    @property
    def output_layer_name(self) -> str:
        """Return the name of the output layer."""
        return f"{self.network_name}output"

    @property
    def flattened_output_shape(self) -> int:
        """Return the flattened output dimensionality."""
        return int(np.prod(self.output_tensor_spec.shape))

    @property
    def model(self) -> tf.keras.Model:
        """
        Return the built Keras model.

        Raises
        ------
        RuntimeError
            If the model has not been built yet.
        """
        if self._model is None:
            raise RuntimeError("Model has not been built yet.")
        return self._model

    def update(self, dataset, epochs: int = 100, verbose: str | int = 0) -> None:
        """
        Trains the model on a dataset.

        Parameters
        ----------
        dataset : Dataset or tf.data.Dataset
            Training data.
        epochs : int, optional
            Number of training epochs.
        verbose : int or str, optional
            Verbosity level for TensorFlow training.
        """
        if isinstance(dataset, Dataset):
            self.model.fit(
                dataset.query_points,
                dataset.observations,
                epochs=epochs,
                verbose=verbose,
            )
        elif isinstance(dataset, tf.data.Dataset):
            self.model.fit(dataset, epochs=epochs, verbose=verbose)
        else:
            raise ValueError(f"Dataset of type {type(dataset)} is not supported.")

    def optimize(self, dataset) -> None:
        """
        Optimizes the model parameters using the provided dataset.

        This is an alias for :meth:`update`.

        Parameters
        ----------
        dataset : Dataset or tf.data.Dataset
            Training data.
        """
        self.update(dataset)


class KerasDropout(KerasPredictor):
    """
    Keras implementation of a Monte Carlo Dropout neural network.

    This class builds a Keras model from a
    :class:`MonteCarloDropoutNetwork` specification and provides
    utilities for MC Dropout inference and serialization.
    """

    def __init__(
        self,
        network: MonteCarloDropoutNetwork,
    ) -> None:
        """
        Initialize the Keras MC Dropout model.

        Parameters
        ----------
        network : MonteCarloDropoutNetwork
            Network architecture specification.
        """
        self._network = network
        # To match KerasEnsemble pattern
        self.num_outputs = network.flattened_output_shape
        self.output_dtype = network.output_tensor_spec.dtype
        self._network = network

        self.input_tensor_spec = network.input_tensor_spec
        self.output_tensor_spec = network.output_tensor_spec

        self.task_type = network.task_type

        self._model: tf.keras.Model = self._build_model()

    def __repr__(self) -> str:
        """Return a string representation of the model."""
        return f"{self.__class__.__name__}({self._network!r})"

    @property
    def model(self) -> tf.keras.Model:
        """Return the built Keras model."""
        return self._model

    def _build_model(self) -> tf.keras.Model:
        """
        Build the Keras model from the network specification.

        Returns
        -------
        tf.keras.Model
            Uncompiled Keras model.
        """
        inputs = tf.keras.Input(
            shape=self.input_tensor_spec.shape,
            dtype=self.input_tensor_spec.dtype,
            name=self._network.input_layer_name,
        )
        x = inputs

        for index, (units, activation, dropout_rate) in enumerate(
            zip(
                self._network.hidden_layers,
                self._network.activations,
                self._network.dropout_rates,
            )
        ):
            layer_name = f"{self._network.network_name}dense_{index}"
            x = tf.keras.layers.Dense(
                units,
                activation=activation,
                name=layer_name,
                dtype=self.input_tensor_spec.dtype.name,
            )(x)
            x = tf.keras.layers.Dropout(dropout_rate)(x)

        if self.task_type is TaskType.REGRESSION:
            outputs = tf.keras.layers.Dense(
                self.output_tensor_spec.shape[-1], activation=None, name="outputs"
            )(x)
        elif self.task_type is TaskType.BINARY_CLASSIFICATION:
            outputs = tf.keras.layers.Dense(1, activation=None, name="logits")(x)
        elif self.task_type is TaskType.MULTIOUTPUT_REGRESSION:
            outputs = tf.keras.layers.Dense(
                self.output_tensor_spec.shape[-1], activation=None, name="outputs"
            )(x)
        elif self.task_type is TaskType.MULTICLASS_CLASSIFICATION:
            outputs = tf.keras.layers.Dense(
                self.output_tensor_spec.shape[-1], activation=None, name="logits"
            )(x)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

        return tf.keras.Model(inputs, outputs)

    def __getstate__(self) -> dict[str, Any]:
        """
        Serialize the model for pickling.

        Returns
        -------
        dict[str, Any]
            Serialized state.
        """

        # When pickling use to_json to save the model.
        state = self.__dict__.copy()
        state["_model"] = self._model.to_json()
        state["_weights"] = self._model.get_weights()

        # Save the history callback (serializing any model)
        if self._model.history:
            history_model = self._model.history.model
            try:
                if history_model is self._model:
                    # no need to serialize the main model, just use a special value instead
                    self._model.history.model = ...
                elif history_model:
                    self._model.history.model = (
                        history_model.to_json(),
                        history_model.get_weights(),
                    )
                state["_history"] = dill.dumps(self._model.history)
            finally:
                self._model.history.model = history_model

        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Restore the model from a serialized state.
        """
        # When unpickling restore the model using model_from_json.
        self.__dict__.update(state)
        self._model = tf.keras.models.model_from_json(state["_model"])
        self._model.set_weights(state["_weights"])

        # Restore the history (including any model it contains)
        if "_history" in state:
            self._model.history = dill.loads(state["_history"])
            if self._model.history.model is ...:
                self._model.history.set_model(self._model)
            elif self._model.history.model:
                model_json, weights = self._model.history.model
                model = tf.keras.models.model_from_json(model_json)
                assert isinstance(
                    model, tf.keras.Model
                ), "Loaded model si not a tf.keras.Model."
                model.set_weights(weights)
                self._model.history.set_model(model)

    def transform_outputs(self, raw_outputs: TensorType) -> TensorType:
        """
        Maps raw model outputs to observation space.
        """
        if self.task_type is TaskType.REGRESSION:
            return raw_outputs

        if self.task_type is TaskType.BINARY_CLASSIFICATION:
            return tf.sigmoid(raw_outputs)

        if self.task_type is TaskType.MULTIOUTPUT_REGRESSION:
            return raw_outputs

        if self.task_type is TaskType.MULTICLASS_CLASSIFICATION:
            return tf.nn.softmax(logits=raw_outputs, axis=-1)

        raise NotImplementedError

    def predict_with_dropout(
        self, query_points: TensorType, num_samples: int = 100
    ) -> tuple[TensorType, TensorType]:
        """
        Perform Monte Carlo Dropout inference.

        Parameters
        ----------
        query_points : TensorType
            Input data of shape ``(n_points, input_dim)``.
        num_samples : int, optional
            Number of stochastic forward passes.

        Returns
        -------
        mean : TensorType
            Predictive mean.
        variance : TensorType
            Predictive variance.
        """
        f_samples = np.stack(
            [self.model(query_points, training=True) for _ in range(num_samples)],
            axis=0,
        )
        mean = tf.math.reduce_mean(f_samples, axis=0)
        variance = tf.math.reduce_variance(f_samples, axis=0)
        return mean, variance


class ProbabilisticNetwork(GaussianNetwork):
    """
    Probabilistic neural network supporting multiple output distributions.

    This class extends Trieste's :class:`GaussianNetwork` to support different
    probabilistic output distributions (e.g. Gaussian, Bernoulli, Categorical)
    depending on the task type.
    """

    def __init__(self, distribution_type: str, *args, **kwargs) -> None:
        """
        Initialize a probabilistic network.

        Parameters
        ----------
        distribution_type : str
            Type of output distribution. Supported values include:
            ``"gaussian"``, ``"normal"``, ``"regression"``,
            ``"bernoulli"``, ``"binary"``, ``"binary_classification"``,
            ``"categorical"``, ``"classification"``.
        *args, **kwargs
            Passed directly to the base :class:`GaussianNetwork` initializer.
        """
        super().__init__(*args, **kwargs)
        self.distribution_type = distribution_type.lower()

    def _gen_multi_output_layer(self, input_tensor: tf.Tensor) -> tf.Tensor:
        """
        Generate a probabilistic output layer for multi-output regression tasks.

        Parameters
        ----------
        input_tensor : tf.Tensor
            Output tensor of the last hidden layer.

        Returns
        -------
        tf.Tensor
            Tensor representing a probabilistic distribution over outputs.

        Raises
        ------
        ValueError
            If the distribution type is unsupported for multi-output networks.
        """
        if self.distribution_type in ["gaussian", "normal", "regression"]:
            dist_layer = (
                tfp.layers.IndependentNormal
                if self._independent
                else MultivariateNormalTriL
            )
            n_params = dist_layer.params_size(self.flattened_output_shape)

            parameter_layer = tf.keras.layers.Dense(
                n_params,
                name=self.network_name + "dense_parameters",
                dtype=input_tensor.dtype.name,
            )(input_tensor)

            distribution = dist_layer(
                self.flattened_output_shape,
                tfp.python.distributions.Distribution.mean,
                name=self.output_layer_name,
                dtype=input_tensor.dtype.name,
            )(parameter_layer)

        else:
            raise ValueError(
                f"Distribution type '{self.distribution_type}' is not supported "
                "for multi-output probabilistic networks."
            )

        return distribution

    def _gen_single_output_layer(self, input_tensor: tf.Tensor) -> tf.Tensor:
        """
        Generate a probabilistic output layer for single-output tasks.

        Parameters
        ----------
        input_tensor : tf.Tensor
            Output tensor of the last hidden layer.

        Returns
        -------
        tf.Tensor
            Tensor representing a probabilistic distribution over outputs.

        Raises
        ------
        ValueError
            If the distribution type is unsupported.
        """
        parameter_layer = tf.keras.layers.Dense(
            2,
            name=self.network_name + "dense_parameters",
            dtype=input_tensor.dtype.name,
        )(input_tensor)

        if self.distribution_type in ["bernoulli", "binary", "binary_classification"]:

            def distribution_fn(inputs: TensorType) -> Distribution:
                return tfp.distributions.Bernoulli(logits=inputs[..., :1])

        elif self.distribution_type in ["categorical", "classification"]:

            def distribution_fn(inputs: TensorType) -> Distribution:
                return tfp.distributions.Categorical(logits=inputs)

        elif self.distribution_type in ["gaussian", "normal", "regression"]:

            def distribution_fn(inputs: TensorType) -> Distribution:
                return tfp.distributions.Normal(
                    loc=inputs[..., :1],
                    scale=tf.math.softplus(inputs[..., 1:]),
                )

        else:
            raise ValueError(
                f"Distribution type '{self.distribution_type}' is not a valid type."
            )

        return tfp.layers.DistributionLambda(
            make_distribution_fn=distribution_fn,
            convert_to_tensor_fn=tfp.distributions.Distribution.mean,
            name=self.output_layer_name,
            dtype=input_tensor.dtype.name,
        )(parameter_layer)

from typing import Union, Tuple, List, Any
import tensorflow as tf
import tensorflow_probability as tfp
from trieste.types import TensorType
import numpy as np
from trieste.data import Dataset
from trieste.models.keras.architectures import GaussianNetwork, MultivariateNormalTriL
import dill


class MonteCarloDropoutNetwork:
    def __init__(self, input_tensor_spec: tf.TensorSpec,
                 output_tensor_spec: tf.TensorSpec,
                 hidden_layers: Union[int, Tuple[int, ...], List[Tuple[int, ...]]] = (32, 32),
                 dropout_rates: Union[float, List[float], Tuple[float, ...]] = 0.0,
                 activations: Union[
                     str, tf.keras.layers.Activation, List[Union[str, tf.keras.layers.Activation]]] = 'relu',
                 network_name: str = ''
                 ):
        """
        A fully connected neural network with dropout for uncertainty estimation (MC Dropout)

        @param input_dim: Dimension of input features
        @param output_dim: Dimension of output
        @param hidden_layers: Tuple defining the number of units in each hidden layer
        @param dropout_rates: Dropout rate used for MC dropout
        @param activations: Activation function for each layer
        """
        self.input_tensor_spec = input_tensor_spec
        self.output_tensor_spec = output_tensor_spec
        self.hidden_layers = hidden_layers
        self.dropout_rates = dropout_rates

        self.activations = activations
        self.network_name = network_name

    @property
    def input_layer_name(self) -> str:
        return self.network_name + 'input'

    @property
    def output_layer_name(self) -> str:
        return self.network_name + 'output'

    @property
    def flattened_output_shape(self) -> int:
        return int(np.prod(self.output_tensor_spec.shape))

    @property
    def model(self) -> tf.keras.Model:
        """Returns built but uncompiled Keras ensemble model."""
        return self._model


    def update(self, dataset, epochs: int = 100, verbose: int = 0):
        """
        Train the model on a given dataset.
        @param dataset: A `Dataset` object from Trieste, containing input-output pairs
        @param epochs: number of epochs to train
        @param verbose: whether to print TF verbose (default none)
        """
        if isinstance(dataset, Dataset):
            self.model.fit(dataset.query_points, dataset.observations, epochs=epochs, verbose=verbose)
        # TODO verify this is valid for Tensorflow Datasets
        else:
            self.model.fit(dataset, epochs=epochs, verbose=verbose)

    def optimize(self, dataset):
        """
        Optimize the model using the provided dataset.

        :@param dataset: A `Dataset` object from Trieste
        """
        self.update(dataset)


class KerasDropout:
    """
    This class builds a MC Dropout neural network using Keras.
     This class mimics KerasEnsemble from trieste.model.keras.architectures.KerasEnsemble andG Gaussian Network
    Model is an instance of KerasDropout
    :class:`~online_neuro.custom_architectures.MonteCarloDropoutNetwork`
    """

    def __init__(
        self,
        network: MonteCarloDropoutNetwork,
    ) -> None:
        """
        :param network: A non-instantiated neural network specifications, one for each member of the
            ensemble. The ensemble will be built using these specifications.
        :raise ValueError: If there are no objects in ``networks`` or we try to create
            a model with networks whose input or output shapes are not the same.
        """

        # To match KerasEnsemble pattern
        self.num_outputs = network.flattened_output_shape
        self.output_dtype = network.output_tensor_spec.dtype
        self._network = network

        self.input_tensor_spec = network.input_tensor_spec
        self.output_tensor_spec = network.output_tensor_spec

        self._model = self._build_model()

    def __repr__(self) -> str:
        """"""
        return f'KerasDropout({self._network!r})'

    @property
    def model(self) -> tf.keras.Model:
        """Returns built but uncompiled Keras dropout model."""
        return self._model

    def _build_model(self):
        """Builds the neural network model."""
        inputs = tf.keras.Input(shape=self.input_tensor_spec.shape,
                                dtype=self.input_tensor_spec.dtype,
                                name=self._network.input_layer_name)
        x = inputs

        for index, (units, activation, dropout_rate) in enumerate(zip(self._network.hidden_layers,
                                                                      self._network.activations,
                                                                      self._network.dropout_rates)):
            layer_name = f'{self._network.network_name}dense_{index}'
            x = tf.keras.layers.Dense(units, activation=activation, name=layer_name, dtype=self.input_tensor_spec.dtype.name)(x)
            x = tf.keras.layers.Dropout(dropout_rate)(x)

        outputs = tf.keras.layers.Dense(self.output_tensor_spec.shape[-1], name=self.output_tensor_spec.dtype.name)(x)
        return tf.keras.Model(inputs, outputs)

    def __getstate__(self) -> dict[str, Any]:
        # When pickling use to_json to save the model.
        state = self.__dict__.copy()
        state['_model'] = self._model.to_json()
        state['_weights'] = self._model.get_weights()

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
                state['_history'] = dill.dumps(self._model.history)
            finally:
                self._model.history.model = history_model

        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        # When unpickling restore the model using model_from_json.
        self.__dict__.update(state)
        self._model = tf.keras.models.model_from_json(state['_model'])
        self._model.set_weights(state['_weights'])

        # Restore the history (including any model it contains)
        if '_history' in state:
            self._model.history = dill.loads(state['_history'])
            if self._model.history.model is ...:
                self._model.history.set_model(self._model)
            elif self._model.history.model:
                model_json, weights = self._model.history.model
                model = tf.keras.models.model_from_json(model_json)
                model.set_weights(weights)
                self._model.history.set_model(model)

    def predict_with_dropout(self, query_points: TensorType, num_samples: int = 100) -> Tuple[TensorType, TensorType]:
        """
        Perform MC Dropout inference to estimate mean and variance.

        @param query_points: Input data (shape: [num_points, input_dim])
        @param num_samples: Number of MC forward passes for uncertainty estimation

        @return: Tuple (mean, variance) of the predictions
        """
        f_samples = np.stack([self.model(query_points, training=True) for _ in range(num_samples)], axis=0)
        mean = tf.math.reduce_mean(f_samples, axis=0)
        variance = tf.math.reduce_variance(f_samples, axis=0)
        return mean, variance

class ProbabilisticNetwork(GaussianNetwork):
    def __init__(self, distribution_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distribution_type = distribution_type.lower()

    def _gen_multi_output_layer(self, input_tensor: tf.Tensor) -> tf.Tensor:
        if self.distribution_type in ['gaussian', 'normal', 'regression']:
            dist_layer = tfp.layers.IndependentNormal if self._independent else MultivariateNormalTriL
            n_params = dist_layer.params_size(self.flattened_output_shape)

            parameter_layer = tf.keras.layers.Dense(
                n_params, name=self.network_name + 'dense_parameters', dtype=input_tensor.dtype.name
            )(input_tensor)

            distribution = dist_layer(
                self.flattened_output_shape,
                tfp.python.distributions.Distribution.mean,
                name=self.output_layer_name,
                dtype=input_tensor.dtype.name,
            )(parameter_layer)

        else:
            msg = f"""Distribution type {self.distribution_type} is not a valid type for modelling
            multi-output probabilistic networks."""
            raise ValueError(msg)

        return distribution

    def _gen_single_output_layer(self, input_tensor: tf.Tensor) -> tf.Tensor:
        parameter_layer = tf.keras.layers.Dense(
            2, name=self.network_name + 'dense_parameters', dtype=input_tensor.dtype.name
        )(input_tensor)

        if self.distribution_type in ['bernoulli', 'binary', 'binary_classification']:
            def distribution_fn(inputs: TensorType) -> tfp.distributions.Distribution:
                return tfp.distributions.Bernoulli(logits=inputs[..., :1])
        elif self.distribution_type in ['categorical', 'classification']:
            def distribution_fn(inputs: TensorType) -> tfp.distributions.Distribution:
                return tfp.distributions.Normal(inputs[..., :1], tf.math.softplus(inputs[..., 1:]))
        elif self.distribution_type in ['gaussian', 'normal', 'regression']:
            def distribution_fn(inputs: TensorType) -> tfp.distributions.Distribution:
                return tfp.distributions.Categorical(logits=inputs)
        else:
            msg = f'Distribution type {self.distribution_type} is not a valid type'
            raise ValueError(msg)

        distribution = tfp.layers.DistributionLambda(
            make_distribution_fn=distribution_fn,
            convert_to_tensor_fn=tfp.distributions.Distribution.mean,
            name=self.output_layer_name,
            dtype=input_tensor.dtype.name,
        )(parameter_layer)

        return distribution

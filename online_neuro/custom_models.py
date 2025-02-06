# Copyright 2025 The IDLAB-imec Contributors
# Notice that partial code is based on The Trieste repository
# https://github.com/secondmind-labs/trieste
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import numpy as np
from trieste.models.interfaces import TrainableProbabilisticModel
from trieste.types import TensorType
from trieste.data import Dataset
from trieste.models.keras.utils import get_tensor_spec_from_data
from typing import Optional, Tuple, Union, List
from trieste.models.keras import (
    DeepEnsemble,
    KerasPredictor,
    build_keras_ensemble,
)
from trieste.models.keras.architectures import GaussianNetwork, KerasEnsemble, MultivariateNormalTriL
import warnings
import tensorflow_probability as tfp
from tensorflow_probability.python.layers.distribution_layer import DistributionLambda, _serialize

class MonteCarloNeuralNetwork:
    def __init__(self, input_dim: int,
                 output_dim: int,
                 hidden_layers: Union[int, Tuple[int, ...], List[Tuple[int, ...]]] = (32, 32),
                 dropout_rates: Union[float, List[float]] = 0.0,
                 activations: Union[str, tf.keras.layers.Activation, List[Union[str, tf.keras.layers.Activation]]] = 'relu',
                 #optimizer: Union[str, tf.keras.optimizers.Optimizer] = 'adam',
                 #loss: Union[str, tf.keras.losses.Loss] = 'mse'
                 ):
        """
        A fully connected neural network with dropout for uncertainty estimation (MC Dropout)

        @param input_dim: Dimension of input features
        @param output_dim: Dimension of output
        @param hidden_layers: Tuple defining the number of units in each hidden layer
        @param dropout_rates: Dropout rate used for MC dropout
        @param activations: Activation function for each layer
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.dropout_rates = dropout_rates
        self.activations = activations
        #self.optimizer = optimizer
        #self.loss = loss
        self._build_model()

    def _build_model(self):
        """Builds the neural network model."""
        inputs = tf.keras.Input(shape=(self.input_dim,))
        x = inputs
        for units, activation, dropout_rate in zip(self.hidden_layers, self.activations, self.dropout_rates):
            x = tf.keras.layers.Dense(units, activation=activation)(x)
            x = tf.keras.layers.Dropout(dropout_rate)(x)

        outputs = tf.keras.layers.Dense(self.output_dim)(x)
        self.model = tf.keras.Model(inputs, outputs)
        #self.model.compile(optimizer=self.optimizer, loss=self.loss)

    def predict(self, query_points: TensorType, num_samples: int = 100) -> Tuple[TensorType, TensorType]:
        """
        Perform MC Dropout inference to estimate mean and variance.

        @param query_points: Input data (shape: [num_points, input_dim])
        @param num_samples: Number of MC forward passes for uncertainty estimation

        @return: Tuple (mean, variance) of the predictions
        """
        f_samples = np.stack([self.model(query_points, training=True) for _ in range(num_samples)], axis=0)
        mean = np.mean(f_samples, axis=0)
        variance = np.var(f_samples, axis=0)
        return mean, variance

    def update(self, dataset, epochs: int = 100, verbose: int =0):
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

        :param dataset: A `Dataset` object from Trieste
        """
        self.update(dataset)


def build_keras_mc_dropout(
        data: Dataset,
        hidden_layers: Union[int, Tuple[int, ...], List[Tuple[int, ...]]] = (32, 32),
        dropout_rates: Union[float, List[float]] = 0.20,
        activations: Union[str, tf.keras.layers.Activation, List[Union[str, tf.keras.layers.Activation]]] = 'relu',
        #optimizer: Union[str, tf.keras.optimizers.Optimizer] = 'adam',
        #loss: Union[str, tf.keras.losses.Loss] = 'mse'
) -> MonteCarloNeuralNetwork:
    """
    Builds a Neural Network with MonterCarlo Dropout enabled compatible with Trieste.
    @param data: Data for training, used for extracting input and output tensor specifications.
    @param hidden_layers: The number of hidden layers in each network or list specifying sizes.
    @param dropout_rates: Dropout rate at each layer.
    @param activations: The activation function in each hidden layer.
    @param optimizer: The optimizer for the NN model.
    @param loss: The loss function to optimize.

    @return: Tensorflow model.
    """
    input_tensor_spec, output_tensor_spec = get_tensor_spec_from_data(data)

    if isinstance(hidden_layers, int):
        warnings.warn("""
        ⚠️ Warning: `hidden_layers` is an int, which might be misinterpreted. Use a tuple (e.g., (64,)) for a single-layer network.
        """)
        hidden_layers = [(hidden_layers,)]
    elif isinstance(hidden_layers, tuple):
        hidden_layers = [hidden_layers]  # Convert single tuple to list for consistency

    num_layers = len(hidden_layers)

    if isinstance(activations, (str, tf.keras.layers.Activation)):
        activations = [activations] * num_layers
    if isinstance(dropout_rates, (float, int)):  # Allow int for dropout = 0
        dropout_rates = [dropout_rates] * num_layers

    model = MonteCarloNeuralNetwork(input_dim=input_tensor_spec,
                                    output_dim=output_tensor_spec,
                                    hidden_layers=hidden_layers,
                                    dropout_rates=dropout_rates,
                                    activations=activations,
                                    #optimizer=optimizer,
                                    #loss=loss
                                    )
    return model


class ProbabilisticNetwork(GaussianNetwork):
    def __init__(self, distribution_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distribution_type = distribution_type.lower()

    def _gen_multi_output_layer(self, input_tensor: tf.Tensor) -> tf.Tensor:
        if self.distribution_type in ['gaussian', 'normal', 'regression']:
            dist_layer = tfp.layers.IndependentNormal if self._independent else MultivariateNormalTriL
            n_params = dist_layer.params_size(self.flattened_output_shape)

            parameter_layer = tf.keras.layers.Dense(
                n_params, name=self.network_name + "dense_parameters", dtype=input_tensor.dtype.name
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
            2, name=self.network_name + "dense_parameters", dtype=input_tensor.dtype.name
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
            msg = f"Distribution type {self.distribution_type} is not a valid type"
            raise ValueError(msg)

        distribution = tfp.layers.DistributionLambda(
            make_distribution_fn=distribution_fn,
            convert_to_tensor_fn=tfp.distributions.Distribution.mean,
            name=self.output_layer_name,
            dtype=input_tensor.dtype.name,
        )(parameter_layer)

        return distribution


def build_keras_ensemble_prob_output(
    data: Dataset,
    ensemble_size: int = 5,
    num_hidden_layers: int = 2,
    units: int = 25,
    activation: Union[str, tf.keras.layers.Activation] = "relu",
    independent_normal: bool = False,
    distribution_type: str = 'Gaussian',
) -> KerasEnsemble:
    """
    Function based on Trieste's version. It allows to specify other type of output distributions,
    such as Gaussian (original), Bernoulli (Binary Classification) and Categorical (Multiclass classification)

    Builds a simple ensemble of neural networks in Keras where each network has the same
    architecture: number of hidden layers, nodes in hidden layers and activation function.

    Default ensemble size and activation function seem to work well in practice, in regression type
    of problems at least. Number of hidden layers and units per layer should be modified according
    to the dataset size and complexity of the function - the default values seem to work well
    for small datasets common in Bayesian optimization. Using the independent normal is relevant
    only if one is modelling multiple output variables, as it simplifies the distribution by
    ignoring correlations between outputs.

    :param data: Data for training, used for extracting input and output tensor specifications.
    :param ensemble_size: The size of the ensemble, that is, the number of base learners or
        individual neural networks in the ensemble.
    :param num_hidden_layers: The number of hidden layers in each network.
    :param units: The number of nodes in each hidden layer.
    :param activation: The activation function in each hidden layer.
    :param independent_normal: If set to `True` then :class:`~tfp.layers.IndependentNormal` layer
        is used as the output layer. This models outputs as independent, only the diagonal
        elements of the covariance matrix are parametrized. If left as the default `False`,
        then :class:`~tfp.layers.MultivariateNormalTriL` layer is used where correlations
        between outputs are learned as well. Note that this is only relevant for multi-output
        models.
    :param distribution_type: Between Normal (default), Bernoulli, and Categorical
    :return: Keras ensemble model.
    """
    input_tensor_spec, output_tensor_spec = get_tensor_spec_from_data(data)

    hidden_layer_args = []
    for _ in range(num_hidden_layers):
        hidden_layer_args.append({"units": units, "activation": activation})

    networks = [
        ProbabilisticNetwork(
            input_tensor_spec=input_tensor_spec,
            output_tensor_spec=output_tensor_spec,
            hidden_layer_args=hidden_layer_args,
            independent=independent_normal,
            distribution_type=distribution_type
        )
        for _ in range(ensemble_size)
    ]
    keras_ensemble = KerasEnsemble(networks)

    return keras_ensemble


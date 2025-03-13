import tensorflow as tf
from typing import Tuple, Union, List

from trieste.data import Dataset
from trieste.models.keras.architectures import KerasEnsemble
from trieste.models.keras.utils import get_tensor_spec_from_data

from .custom_architectures import MonteCarloDropoutNetwork, KerasDropout, ProbabilisticNetwork
import warnings


def build_keras_mc_dropout(
        data: Dataset,
        hidden_layers: int | Tuple[int, ...] | List[Tuple[int, ...]] = (32, 32),
        dropout_rates: float | List[float] = 0.20,
        activations: str | tf.keras.layers.Activation | List[ str | tf.keras.layers.Activation] = 'relu'
) -> KerasDropout:
    """
    Builds a Neural Network with MonterCarlo Dropout enabled compatible with Trieste.
    @param data: Data for training, used for extracting input and output tensor specifications.
    @param hidden_layers: The number of hidden layers in each network or list specifying sizes.
    @param dropout_rates: Dropout rate at each layer.
    @param activations: The activation function in each hidden layer.

    @return: Tensorflow model.
    """
    input_tensor_spec, output_tensor_spec = get_tensor_spec_from_data(data)

    if isinstance(hidden_layers, int):
        warnings.warn("""
        ⚠️ Warning: `hidden_layers` is an int, which might be misinterpreted. Use a tuple (e.g., (64,)) for a single-layer network.
        """)
        hidden_layers = [(hidden_layers,)]
    elif isinstance(hidden_layers, tuple):
        hidden_layers = [(x,) if isinstance(x, int) else x for x in hidden_layers]
    else:
        hidden_layers = hidden_layers

    num_layers = len(hidden_layers)

    if isinstance(activations, (str, tf.keras.layers.Activation)):
        activations = [activations] * num_layers

    if isinstance(dropout_rates, float):  # Allow int for dropout = 0
        dropout_rates = [dropout_rates] * num_layers
    elif isinstance(dropout_rates, tuple):
        dropout_rates = list(dropout_rates)

    mc_network = MonteCarloDropoutNetwork(input_tensor_spec=input_tensor_spec,
                                          output_tensor_spec=output_tensor_spec,
                                          hidden_layers=hidden_layers,
                                          dropout_rates=dropout_rates,
                                          activations=activations
                                          )
    model = KerasDropout(mc_network)
    return model


def build_keras_ensemble_prob_output(
    data: Dataset,
    ensemble_size: int = 5,
    num_hidden_layers: int = 2,
    units: int = 25,
    activation: Union[str, tf.keras.layers.Activation] = 'relu',
    independent_normal: bool = False,
    distribution_type: str = 'Gaussian',
) -> KerasEnsemble:
    """
    TODO, change nu
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
        hidden_layer_args.append({'units': units, 'activation': activation})

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

from typing import List

import tensorflow as tf
from trieste.data import Dataset
from trieste.models.keras.architectures import KerasEnsemble
from trieste.models.keras.utils import get_tensor_spec_from_data

from .custom_architectures import (
    TASK_ALIASES,
    KerasDropout,
    MonteCarloDropoutNetwork,
    ProbabilisticNetwork,
    TaskType,
)


def build_keras_mc_dropout(
    data: Dataset,
    hidden_layers: int | tuple[int, ...] | list[int] = (32, 32),
    dropout_rates: float | tuple[float, ...] | list[float] = 0.2,
    activations: (
        str | tf.keras.layers.Activation | List[str | tf.keras.layers.Activation]
    ) = "relu",
    task_type: TaskType | str = TaskType.REGRESSION,
) -> KerasDropout:
    """
    Build a Monte Carlo Dropout neural network compatible with Trieste.

    This function constructs a fully connected neural network with dropout layers
    that remain active at inference time, enabling uncertainty estimation via
    Monte Carlo sampling.

    Parameters
    ----------
    data : Dataset
        Trieste dataset used to infer input and output tensor specifications.
    hidden_layers : int or sequence of int, optional
        Number of units in each hidden layer. If an integer is provided, a single
        hidden layer is created.
    dropout_rates : float or sequence of float, optional
        Dropout rate applied after each hidden layer.
    activations : str, Activation, or sequence, optional
        Activation function(s) for the hidden layers.

    Returns
    -------
    KerasDropout
        A KerasDropout model ready for compilation and training.

    Raises
    ------
    ValueError
        If the lengths of ``hidden_layers``, ``dropout_rates``, and ``activations``
        are incompatible.
    """

    input_tensor_spec, output_tensor_spec = get_tensor_spec_from_data(data)

    if isinstance(hidden_layers, int):
        hidden_layers = [hidden_layers]
    else:
        hidden_layers = list(hidden_layers)

    num_layers = len(hidden_layers)

    if isinstance(activations, (str, tf.keras.layers.Activation)):
        activations = [activations] * num_layers
    else:
        activations = list(activations)

    if isinstance(dropout_rates, (int, float)):
        dropout_rates = [float(dropout_rates)] * num_layers
    else:
        dropout_rates = list(dropout_rates)

    if not all(0.0 <= d < 1.0 for d in dropout_rates):
        raise ValueError("Dropout rates must be in [0, 1).")

    if not (len(hidden_layers) == len(activations) == len(dropout_rates)):
        raise ValueError(
            "Lengths of hidden_layers, activations, and dropout_rates must match."
        )

    if isinstance(task_type, str):
        key = task_type.lower()
        task_type = TASK_ALIASES[key]

    mc_network = MonteCarloDropoutNetwork(
        input_tensor_spec=input_tensor_spec,
        output_tensor_spec=output_tensor_spec,
        hidden_layers=hidden_layers,
        dropout_rates=dropout_rates,
        activations=activations,
        task_type=task_type,
    )
    model = KerasDropout(mc_network)
    return model


def build_keras_ensemble_prob_output(
    data: Dataset,
    ensemble_size: int = 5,
    num_hidden_layers: int = 2,
    units: int = 25,
    activation: str | tf.keras.layers.Activation = "relu",
    independent_normal: bool = False,
    distribution_type: str = "Gaussian",
) -> KerasEnsemble:
    """
    Build a probabilistic Keras ensemble model compatible with Trieste.

    This function constructs an ensemble of neural networks with identical
    architectures and probabilistic output layers. The output distribution
    can be Gaussian (default), Bernoulli (binary classification), or
    Categorical (multi-class classification).

    The ensemble approach improves predictive uncertainty estimation by
    aggregating multiple independently initialized models.

    Parameters
    ----------
    data : Dataset
        Trieste dataset used to infer input and output tensor specifications.
    ensemble_size : int, optional
        Number of neural networks in the ensemble.
    num_hidden_layers : int, optional
        Number of hidden layers in each network.
    units : int, optional
        Number of units in each hidden layer.
    activation : str or tf.keras.layers.Activation, optional
        Activation function used in each hidden layer.
    independent_normal : bool, optional
        If ``True``, model multi-output regression using independent normal
        distributions (diagonal covariance). If ``False``, a full covariance
        multivariate normal distribution is used.
    distribution_type : str, optional
        Output distribution type. Supported values include:
        ``"gaussian"``, ``"normal"``, ``"regression"``,
        ``"bernoulli"``, ``"binary"``,
        ``"categorical"``, ``"classification"``.

    Returns
    -------
    KerasEnsemble
        A probabilistic Keras ensemble model compatible with Trieste.

    Raises
    ------
    ValueError
        If ``ensemble_size`` or ``num_hidden_layers`` is less than 1.
    """
    if ensemble_size < 1:
        raise ValueError("ensemble_size must be at least 1.")

    if num_hidden_layers < 1:
        raise ValueError("num_hidden_layers must be at least 1.")

    input_tensor_spec, output_tensor_spec = get_tensor_spec_from_data(data)

    hidden_layer_args = [
        {"units": units, "activation": activation} for _ in range(num_hidden_layers)
    ]

    networks = [
        ProbabilisticNetwork(
            input_tensor_spec=input_tensor_spec,
            output_tensor_spec=output_tensor_spec,
            hidden_layer_args=hidden_layer_args,
            independent=independent_normal,
            distribution_type=distribution_type,
        )
        for _ in range(ensemble_size)
    ]

    return KerasEnsemble(networks)

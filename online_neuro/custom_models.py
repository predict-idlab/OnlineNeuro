# /online_neuro/custom_models.py
import re
import warnings
from typing import Any, Dict, Mapping, Optional

import dill
import keras
import tensorflow as tf
import tensorflow_probability.python.distributions as tfd
from check_shapes import inherit_check_shapes
from keras.losses import MeanSquaredError
from tensorflow.python.keras.callbacks import Callback
from trieste import logging
from trieste.data import Dataset
from trieste.models.interfaces import TrajectorySampler
from trieste.models.keras.architectures import MultivariateNormalTriL
from trieste.models.keras.models import DeepEnsemble
from trieste.models.keras.utils import sample_with_replacement
from trieste.models.optimizer import KerasOptimizer
from trieste.models.utils import write_summary_data_based_metrics
from trieste.types import TensorType
from trieste.utils.misc import flatten_leading_dims

from .custom_architectures import KerasDropout

# Copyright 2025 The IDLAB-imec Contributors
# Notice that partial code is based on The Trieste repository
# https://github.com/secondmind-labs/trieste


class DeepDropout(DeepEnsemble):
    """
    A child from :class:`~trieste.model.TrainableProbabilisticModel` wrapper for neural networks
    which uses MC dropout built with Keras. (Yarin Gal 2015 https://arxiv.org/abs/1506.02142).

    Currently, we do not support setting up the model with dictionary config.
    """

    def __init__(
        self,
        model: KerasDropout,
        optimizer: Optional[KerasOptimizer] = None,
        bootstrap: bool = False,
        diversify: bool = False,
        continuous_optimisation: bool = True,
        compile_args: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        :param model: A KerasDropout model with probabilistic output based on MC dropout during
            inference.
            model has to be built but not compiled.
        :param optimizer: The optimizer wrapper with necessary specifications for compiling and
            training the model. Defaults to :class:`~trieste.models.optimizer.KerasOptimizer` with
            :class:`~tf.optimizers.Adam` optimizer, negative log likelihood loss, mean squared
            error metric and a dictionary of default arguments for Keras `fit` method: 3000 epochs,
            batch size 16, early stopping callback with patience of 50, and verbose 0.
            See https://keras.io/api/models/model_training_apis/#fit-method for a list of possible
            arguments.
        :param bootstrap: Sample with replacement data for training each network in the ensemble.
            By default set to `False`.
        :param diversify: Whether to use quantiles from the approximate Gaussian distribution of
            the ensemble as trajectories instead of mean predictions when calling
            :meth:`trajectory_sampler`. This mode can be used to increase the diversity
            in case of optimizing very large batches of trajectories. By
            default set to `False`.
        :param continuous_optimisation: If True (default), the optimizer will keep track of the
            number of epochs across BO iterations and use this number as initial_epoch. This is
            essential to allow monitoring of model training across BO iterations.
        :param compile_args: Keyword arguments to pass to the ``compile`` method of the
            Keras model (:class:`~tf.keras.Model`).
            See https://keras.io/api/models/model_training_apis/#compile-method for a
            list of possible arguments. The ``optimizer``, ``loss`` and ``metrics`` arguments
            must not be included.
        :raise ValueError: If ``model`` is not an instance of
            :class:`~trieste.models.keras.KerasEnsemble`, or ensemble has less than two base
            learners (networks), or `compile_args` contains disallowed arguments.
        """
        if optimizer is None:
            optimizer = KerasOptimizer(tf.optimizers.Adam())
        self._optimizer = optimizer

        if compile_args is None:
            compile_args = {}

        if not isinstance(optimizer.optimizer, tf.optimizers.Optimizer):
            raise ValueError(
                f"Optimizer for `KerasPredictor` models must be an instance of a "
                f"`tf.optimizers.Optimizer`, received {type(optimizer.optimizer)} instead."
            )

        if not {"optimizer", "loss", "metrics"}.isdisjoint(compile_args):
            raise ValueError(
                "optimizer, loss and metrics arguments must not be included in compile_args."
            )

        if not self.optimizer.fit_args:
            self.optimizer.fit_args = {
                "verbose": 0,
                "epochs": 100,
                "batch_size": 16,
                "callbacks": [
                    tf.keras.callbacks.EarlyStopping(
                        monitor="loss", patience=10, restore_best_weights=True
                    )
                ],
            }

        if self.optimizer.loss is None:
            warnings.warn(
                """⚠️ Warning: `optimizer.loss` is None. Defaulting to MeanSquaredError."""
            )
            self.optimizer.loss = MeanSquaredError

        if self.optimizer.metrics is None:
            warnings.warn(
                """⚠️ Warning: `optimizer.metrics` is None. Defaulting to MSE."""
            )
            self.optimizer.metrics = ["mse"]

        model.model.compile(
            optimizer=self.optimizer.optimizer,
            loss=self.optimizer.loss,
            metrics=self.optimizer.metrics,
            **compile_args,
        )

        if not isinstance(
            self.optimizer.optimizer.lr,
            tf.keras.optimizers.schedules.LearningRateSchedule,
        ):
            self.original_lr = self.optimizer.optimizer.lr.numpy()
        self._absolute_epochs = 0
        self._continuous_optimisation = continuous_optimisation

        self._model = model
        self._bootstrap = bootstrap
        self._diversify = diversify

    def __repr__(self) -> str:
        """"""
        return (
            f"DeepDropout({self.model!r}, {self.optimizer!r}, {self._bootstrap!r}, "
            f"{self._diversify!r})"
        )

    @property
    def ensemble_size(self) -> int:
        """
        Deep Dropout is not an ensemble as it only uses one network which can be viewed
        as multiple networks.
        @todo
        However, Trieste uses this property and some extra work is needed to correctly adjust this class
        to full compatibility with Trieste.
        """
        raise NotImplementedError

    def prepare_dataset(
        self, dataset: Dataset
    ) -> tuple[Dict[str, TensorType], Dict[str, TensorType]]:
        """
        Transform ``dataset`` into inputs and outputs with correct names that can be used for
        training the :class:`KerasDropout` model.

        If ``bootstrap`` argument in the :class:`~trieste.models.keras.DeepEnsemble` is set to
        `True`, data will be additionally sampled with replacement, independently for
        each network in the ensemble.

        :param dataset: A dataset with ``query_points`` and ``observations`` tensors.
        :return: A dictionary with input data and a dictionary with output data.
        """
        if self._bootstrap:
            resampled_data = sample_with_replacement(dataset)
        else:
            resampled_data = dataset

        inputs, outputs = resampled_data.astuple()

        return inputs, outputs

    def prepare_query_points(self, query_points: TensorType) -> Dict[str, TensorType]:
        """
        Transform ``query_points`` into inputs with correct names that can be used for
        predicting with the model.

        :param query_points: A tensor with ``query_points``.
        :return: A dictionary with query_points prepared for predictions.
        """
        raise NotImplementedError

    def predict_estimation(
        self, query_points: TensorType, num_samples=100
    ) -> TensorType:
        return self._model.predict_with_dropout(query_points, num_samples=num_samples)

    def ensemble_distributions(
        self, query_points: TensorType
    ) -> tuple[tfd.Distribution, ...]:
        """
        Return distributions for each member of the ensemble.

        :param query_points: The points at which to return distributions.
        :return: The distributions for the observations at the specified
            ``query_points`` for each member of the ensemble.
        """
        raise NotImplementedError

    @inherit_check_shapes
    def predict(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        r"""
        Returns mean and variance at ``query_points`` for the dropout method.

        TODO find the explanation of this method. i.e. document it similar to Trieste's
         equivalent of Ensemble.

        :param query_points: The points at which to make predictions.
        :return: The predicted mean and variance of the observations at the specified
            ``query_points``.
        """
        # handle leading batch dimensions, while still allowing `Functional` to
        # "allow (None,) and (None, 1) Tensors to be passed interchangeably"
        input_dims = min(
            len(query_points.shape), len(self.model.layers[0].input_shape[0])
        )
        flat_x, unflatten = flatten_leading_dims(query_points, output_dims=input_dims)
        mean, variance = self.predict_estimation(flat_x)

        return unflatten(mean), unflatten(variance)

    @property
    def optimizer(self) -> KerasOptimizer:
        """The optimizer wrapper for training the model."""
        return self._optimizer

    @property
    def dtype(self) -> tf.DType:
        """The prediction dtype."""
        return self._model.output_dtype

    def predict_ensemble(
        self, query_points: TensorType
    ) -> tuple[TensorType, TensorType]:
        """
        Returns mean and variance at ``query_points`` for each member of the ensemble. First tensor
        is the mean and second is the variance, where each has shape [..., M, N, 1], where M is
        the ``ensemble_size``.

        This method assumes that the final layer in each member of the ensemble is
        probabilistic, an instance of :class:`¬tfp.distributions.Distribution`, in particular
        `mean` and `variance` methods should be available.

        :param query_points: The points at which to make predictions.
        :return: The predicted mean and variance of the observations at the specified
            ``query_points`` for each member of the ensemble.
        """
        # ensemble_distributions = self.ensemble_distributions(query_points)
        # predicted_means = tf.convert_to_tensor([dist.mean() for dist in ensemble_distributions])
        # predicted_vars = tf.convert_to_tensor([dist.variance() for dist in ensemble_distributions])
        # return predicted_means, predicted_vars
        raise NotImplementedError

    def sample_ensemble(self, query_points: TensorType, num_samples: int) -> TensorType:
        """
        Return ``num_samples`` samples at ``query_points``. Each sample is taken from a Gaussian
        distribution given by the predicted mean and variance of a randomly chosen network in the
        ensemble. This avoids using the Gaussian mixture approximation and samples directly from
        individual Gaussian distributions given by each network in the ensemble.

        :param query_points: The points at which to sample, with shape [..., N, D].
        :param num_samples: The number of samples at each point.
        :return: The samples. For a predictive distribution with event shape E, this has shape
            [..., S, N] + E, where S is the number of samples.
        """
        raise NotImplementedError

    def trajectory_sampler(self) -> TrajectorySampler[DeepEnsemble]:
        """
        Return a trajectory sampler. For :class:`DeepEnsemble`, we use an ensemble
        sampler that randomly picks a network from the ensemble and uses its predicted means
        for generating a trajectory, or optionally randomly sampled quantiles rather than means.

        :return: The trajectory sampler.
        """
        # TODO Maybe implement?
        # return DeepEnsembleTrajectorySampler(self, self._diversify)
        raise NotImplementedError

    def update(self, dataset: Dataset) -> None:
        """
        Neural networks are parametric models and do not need to update data.
        `TrainableProbabilisticModel` interface, however, requires an update method, so
        here we simply pass the execution.
        """
        return

    def log(self, dataset: Optional[Dataset] = None) -> None:
        """
        Log model training information at a given optimization step to the Tensorboard.
        We log several summary statistics of losses and metrics given in ``fit_args`` to
        ``optimizer`` (final, difference between inital and final loss, min and max). We also log
        epoch statistics, but as histograms, rather than time series. We also log several training
        data based metrics, such as root mean square error between predictions and observations,
        and several others.

        We do not log statistics of individual models in the ensemble unless specifically switched
        on with ``trieste.logging.set_summary_filter(lambda name: True)``.

        For custom logs user will need to subclass the model and overwrite this method.

        :param dataset: Optional data that can be used to log additional data-based model summaries.
        """
        summary_writer = logging.get_tensorboard_writer()
        if summary_writer:
            with summary_writer.as_default(step=logging.get_step_number()):
                logging.scalar("epochs/num_epochs", len(self.model.history.epoch))
                for k, v in self.model.history.history.items():
                    KEY_SPLITTER = {
                        # map history keys to prefix and suffix
                        "loss": ("loss", ""),
                        r"(?P<model>model_\d+)_output_loss": ("loss", r"_\g<model>"),
                        r"(?P<model>model_\d+)_output_(?P<metric>.+)": (
                            r"\g<metric>",
                            r"_\g<model>",
                        ),
                    }
                    for pattern, (pre_sub, post_sub) in KEY_SPLITTER.items():
                        if re.match(pattern, k):
                            pre = re.sub(pattern, pre_sub, k)
                            post = re.sub(pattern, post_sub, k)
                            break
                    else:
                        # unrecognised history key; ignore
                        continue
                    if "model" in post:
                        if not logging.include_summary("_dropout"):
                            break
                        pre = pre + "/_dropout"
                    logging.histogram(f"{pre}/epoch{post}", lambda: v)
                    logging.scalar(f"{pre}/final{post}", lambda: v[-1])
                    logging.scalar(f"{pre}/diff{post}", lambda: v[0] - v[-1])
                    logging.scalar(f"{pre}/min{post}", lambda: tf.reduce_min(v))
                    logging.scalar(f"{pre}/max{post}", lambda: tf.reduce_max(v))
                if dataset:
                    write_summary_data_based_metrics(
                        dataset=dataset, model=self, prefix="training_"
                    )
                    if logging.include_summary("_dropout"):
                        predict_ensemble_variance = self.predict(dataset.query_points)[
                            1
                        ]
                        for i in range(predict_ensemble_variance.shape[0]):
                            logging.histogram(
                                f"variance/_dropout/predict_variance_model_{i}",
                                predict_ensemble_variance[i, ...],
                            )
                            logging.scalar(
                                f"variance/_dropout/predict_variance_mean_model_{i}",
                                tf.reduce_mean(predict_ensemble_variance[i, ...]),
                            )

    def __getstate__(self) -> dict[str, Any]:
        # use to_json and get_weights to save any optimizer fit_arg callback models
        state = self.__dict__.copy()
        if self._optimizer:
            callbacks: list[Callback] = self._optimizer.fit_args.get("callbacks", [])
            saved_models: list[KerasOptimizer] = []
            tensorboard_writers: list[dict[str, Any]] = []
            try:
                for callback in callbacks:
                    # serialize the callback models before pickling the optimizer
                    saved_models.append(callback.model)
                    if callback.model is self.model:
                        # no need to serialize the main model, just use a special value instead
                        callback.model = ...
                    elif callback.model:
                        callback.model = (
                            callback.model.to_json(),
                            callback.model.get_weights(),
                        )
                    # don't pickle tensorboard writers either; they'll be recreated when needed
                    if isinstance(callback, tf.keras.callbacks.TensorBoard):
                        tensorboard_writers.append(callback._writers)
                        callback._writers = {}
                state["_optimizer"] = dill.dumps(state["_optimizer"])
            except Exception as e:
                raise NotImplementedError(
                    "Failed to copy DeepDropout optimizer due to unsupported callbacks."
                ) from e
            finally:
                # revert original state, even if the pickling failed
                for callback, model in zip(callbacks, saved_models):
                    callback.model = model
                for callback, writers in zip(
                    (
                        cb
                        for cb in callbacks
                        if isinstance(cb, tf.keras.callbacks.TensorBoard)
                    ),
                    tensorboard_writers,
                ):
                    callback._writers = writers

        # don't serialize any history optimization result
        if isinstance(state.get("_last_optimization_result"), keras.callbacks.History):
            state["_last_optimization_result"] = ...

        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        # Restore optimizer and callback models after depickling, and recompile.
        self.__dict__.update(state)

        # Unpickle the optimizer, and restore all the callback models
        self._optimizer = dill.loads(self._optimizer)
        for callback in self._optimizer.fit_args.get("callbacks", []):
            if callback.model is ...:
                callback.set_model(self.model)
            elif callback.model:
                model_json, weights = callback.model
                model = tf.keras.models.model_from_json(
                    model_json,
                    custom_objects={"MultivariateNormalTriL": MultivariateNormalTriL},
                )
                model.set_weights(weights)
                callback.set_model(model)

        # Recompile the model
        self.model.compile(
            self.optimizer.optimizer,
            loss=self.optimizer.loss,
            metrics=self.optimizer.metrics,
        )

        # recover optimization result if necessary (and possible)
        if state.get("_last_optimization_result") is ...:
            self._last_optimization_result = getattr(self.model, "history")

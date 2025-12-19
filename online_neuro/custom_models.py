# /online_neuro/custom_models.py
import warnings
from typing import Any

import dill
import tensorflow as tf
from keras.losses import MeanSquaredError
from trieste.data import Dataset
from trieste.models.interfaces import (
    EncodedSupportsPredictY,
    EncodedTrainableProbabilisticModel,
    HasTrajectorySampler,
    TrajectoryFunction,
    TrajectorySampler,
)
from trieste.models.keras.interface import KerasPredictor
from trieste.models.optimizer import KerasOptimizer
from trieste.space import EncoderFunction
from trieste.types import TensorType

from .custom_architectures import KerasDropout, TaskType


class MCDropoutTrajectorySampler(TrajectorySampler):
    """
    Trajectory sampler for the MCDropout model.
    Returns a callable corresponding to the mean prediction of a single stochastic pass.
    """

    def __init__(self, model: "MCDropout"):
        self._model = model

    def sample_trajectory(self) -> TensorType:
        # Request a callable that performs a single stochastic prediction (mean)
        return self._model.predict_single_trajectory_mean_encoded()

    def update_trajectory(self, trajectory: TrajectoryFunction) -> None:
        pass


class MCDropout(
    KerasPredictor,
    EncodedTrainableProbabilisticModel,
    HasTrajectorySampler,
    EncodedSupportsPredictY,
):
    """
    Monte Carlo (MC) Dropout probabilistic model wrapper for Keras networks compatible with
    Trieste.

    This class implements a single neural network using Monte Carlo Dropout for uncertainty
    estimation (Gal & Ghahramani, 2015, https://arxiv.org/abs/1506.02142).

    Unlike :class:`DeepEnsemble`, this model does not maintain multiple networks.
    Instead, uncertainty is estimated by enabling dropout at inference time.
    """

    def __init__(
        self,
        model: KerasDropout,  # Use a generic interface for the single net
        optimizer: KerasOptimizer | None = None,
        mc_samples: int = 50,  # Number of stochastic passes to simulate the ensemble
        bootstrap: bool = False,
        diversify: bool = False,
        continuous_optimisation: bool = True,
        compile_args: dict[str, Any] | None = None,
        encoder: EncoderFunction | None = None,
    ) -> None:
        super().__init__(optimizer, encoder)

        if compile_args is None:
            compile_args = {}

        if not isinstance(optimizer.optimizer, tf.keras.optimizers.Optimizer):
            raise ValueError(
                f"Optimizer for `KerasPredictor` models must be an instance of a "
                f"`tf.optimizers.Optimizer`, received {type(optimizer.optimizer)} instead."
            )

        if not {"optimizer", "loss", "metrics"}.isdisjoint(compile_args):
            raise ValueError(
                "optimizer, loss and metrics arguments must not be included in compile_args."
            )

        # 1. Setup standard KerasOptimizer defaults (same as Trieste's DeepEnsemble)
        if not self.optimizer.fit_args:
            self.optimizer.fit_args = {
                "verbose": 0,
                "epochs": 3000,
                "batch_size": 16,
                "callbacks": [
                    tf.keras.callbacks.EarlyStopping(
                        monitor="loss", patience=50, restore_best_weights=True
                    )
                ],
            }

        if self.optimizer.loss is None:
            warnings.warn(
                """ Warning: `optimizer.loss` is None. Defaulting to MeanSquaredError."""
            )
            self.optimizer.loss = MeanSquaredError

        if self.optimizer.metrics is None:
            warnings.warn(
                """Warning: `optimizer.metrics` is None. Defaulting to MSE."""
            )
            self.optimizer.metrics = ["mse"]

        # 2. Compile the SINGLE Keras model (loss/metrics are not replicated)
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
            # Store original LR for reset functionality
            self.original_lr = self.optimizer.optimizer.lr.numpy()

        self._model = model
        self._mc_samples = mc_samples
        self._continuous_optimisation = continuous_optimisation
        self._absolute_epochs = 0
        self._encoder = encoder

        # Retained for Trajectory Sampler compatibility, not used
        self._bootstrap = bootstrap
        self._diversify = diversify

    def __repr__(self) -> str:
        return f"MCDropout({self._model!r}, mc_samples={self._mc_samples!r}, {self.optimizer!r})"

    @property
    def model(self) -> tf.keras.Model:
        """Returns compiled Keras model."""
        return self._model.model

    @property
    def ensemble_size(self) -> int:
        """Returns the size of the simulated ensemble (MC samples)."""
        return self._mc_samples

    @property
    def num_outputs(self) -> int:
        """Returns the number of outputs trained on by the network."""
        return self._model.num_outputs

    @property
    def dtype(self) -> tf.DType:
        """The prediction dtype."""
        return self._model.output_dtype

    def prepare_dataset(
        self, dataset: Dataset, do_not_bootstrap: bool = False
    ) -> tuple[TensorType, TensorType]:
        """
        Placeholder kept to follow DeepEnsemble.
        in MC the data variation is done by Dropout and not
        by bootstrapping.

        """
        return dataset.query_points, dataset.observations

    def prepare_query_points(self, query_points: TensorType) -> TensorType:
        """
        Placeholder kept to follow DeepEnsemble.

        Returns the query points tensor directly.
        """
        return query_points

    # --- Prediction Methods ---
    def get_raw_mc_samples_encoded(self, query_points: TensorType) -> TensorType:
        """
        Performs M stochastic forward passes and returns the raw outputs.
        Shape: [M, N, 1] (or [M, batch..., N, 1] after unflattening)
        """
        x_transformed = self.prepare_query_points(query_points)

        # We use the raw Keras call and rely on training=True to enable dropout
        raw_samples = tf.stack(
            [self.model(x_transformed, training=True) for _ in range(self._mc_samples)],
            axis=-3,  # Stack along the ensemble dimension
        )
        # Transform the outputs based on the task
        return self._model.transform_outputs(raw_samples)

    def predict_ensemble_encoded(
        self, query_points: TensorType
    ) -> tuple[TensorType, TensorType]:
        """
        Simulates the DeepEnsemble output: M means (raw predictions) and M zero variances.
        Shape: [..., M, N, 1] for both outputs.
        """
        # 1. Get M raw predictions (these serve as the ensemble means mu_m)
        predicted_means = self.get_raw_mc_samples_encoded(query_points)

        # 2. Aleatoric Variance (sigma^2_m) is zero, as the model outputs point predictions
        predicted_vars = tf.zeros_like(predicted_means, dtype=predicted_means.dtype)

        return predicted_means, predicted_vars

    # --- Derived Prediction Methods (Identical to DeepEnsemble) ---

    def predict_encoded(
        self, query_points: TensorType
    ) -> tuple[TensorType, TensorType]:
        ensemble_means, _ = self.predict_ensemble_encoded(query_points)
        predicted_means = tf.math.reduce_mean(ensemble_means, axis=-3)
        epistemic_variance = tf.math.reduce_variance(ensemble_means, axis=-3)

        # Epistemic variance, since aleatoric is zero
        return predicted_means, epistemic_variance

    def predict_y_encoded(
        self, query_points: TensorType
    ) -> tuple[TensorType, TensorType]:
        # Total variance. Since Aleatoric variance is 0 (from predict_ensemble_encoded),
        # Total variance = Epistemic variance.
        predicted_means, epistemic_variance = self.predict_encoded(query_points)

        return predicted_means, epistemic_variance

    def predict_noise_encoded(
        self, query_points: TensorType
    ) -> tuple[TensorType, TensorType]:
        # Aleatoric noise prediction (should return 0, 0)
        _, ensemble_vars = self.predict_ensemble_encoded(query_points)
        aleatoric_variance_mean = tf.math.reduce_mean(
            ensemble_vars, axis=-3
        )  # Should be 0
        aleatoric_variance_variance = tf.math.reduce_variance(
            ensemble_vars, axis=-3
        )  # Should be 0
        return aleatoric_variance_mean, aleatoric_variance_variance

    def predictive_entropy(self, query_points):
        probs, _ = self.predict_y(query_points)

        eps = 1e-7
        if self._model.task_type is TaskType.BINARY_CLASSIFICATION:
            return -(
                probs * tf.math.log(probs + eps)
                + (1 - probs) * tf.math.log(1 - probs + eps)
            )

        if self._model.task_type is TaskType.MULTICLASS_CLASSIFICATION:
            return -tf.reduce_sum(probs * tf.math.log(probs + eps), axis=-1)

    # --- Sampling and Trajectory ---
    def predict_single_trajectory_mean_encoded(self) -> TensorType:
        """
        Returns a callable trajectory predictor function (f(x) -> mu_single_stochastic_pass).
        """

        def trajectory_prediction_fn(x_encoded: TensorType) -> TensorType:
            return self.model(x_encoded, training=True)

        return trajectory_prediction_fn

    def trajectory_sampler(self) -> TrajectorySampler:
        return MCDropoutTrajectorySampler(self)

    # --- Optimization and Logging (Boilerplate largely reused) ---

    def update_encoded(self, dataset: Dataset) -> None:
        """Pass, as model is parametric."""
        return

    def optimize_encoded(self, dataset: Dataset) -> tf.keras.callbacks.History:
        # Optimization logic is identical to DeepEnsemble/DeepDropout (use Keras fit)
        fit_args = dict(self.optimizer.fit_args)

        if "epochs" in fit_args:
            fit_args["epochs"] = fit_args["epochs"] + self._absolute_epochs

        x, y = self.prepare_dataset(dataset)
        tf_train_dataset = self._build_tf_dataset(x, y)

        history = self.model.fit(
            tf_train_dataset,
            **fit_args,
            initial_epoch=self._absolute_epochs,
        )
        if self._continuous_optimisation:
            self._absolute_epochs = self._absolute_epochs + len(history.history["loss"])

        if not isinstance(
            self.optimizer.optimizer.lr,
            tf.keras.optimizers.schedules.LearningRateSchedule,
        ):
            self.optimizer.optimizer.lr.assign(self.original_lr)

        return history

    def _build_tf_dataset(self, x: TensorType, y: TensorType) -> tf.data.Dataset:
        """Builds a tf.data.Dataset from simple X, Y tensors."""
        tf_dataset = tf.data.Dataset.from_tensor_slices((x, y))

        if "steps_per_epoch" in self.optimizer.fit_args:
            tf_dataset = tf_dataset.prefetch(tf.data.experimental.AUTOTUNE).repeat()

        if "batch_size" in self.optimizer.fit_args:
            batch_size = self.optimizer.fit_args["batch_size"]
            tf_dataset = tf_dataset.batch(batch_size)

        return tf_dataset

    # --- Serialization---
    def __getstate__(self) -> dict[str, Any]:
        # Implementation adapted from DeepEnsemble to handle single model serialization
        state = self.__dict__.copy()

        # NOTE: Must manually handle serialization of callback models and optimizer state using dill
        # (Exact implementation requires handling of `self.optimizer` and `self.model` status)

        # Simplified serialization boilerplate (assuming necessary imports like MultivariateNormalTriL)
        if self._optimizer:
            # Placeholder for complex callback serialization logic from DeepEnsemble
            try:
                state["_optimizer"] = dill.dumps(state["_optimizer"])
            except Exception as e:
                raise NotImplementedError(
                    "Failed to copy MCDropout optimizer due to unsupported callbacks."
                ) from e

        # don't serialize any history optimization result
        if isinstance(
            state.get("_last_optimization_result"), tf.keras.callbacks.History
        ):
            state["_last_optimization_result"] = ...

        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

        # Restore optimizer and callback models after depickling
        self._optimizer = dill.loads(self._optimizer)
        # Restore callback models (omitted complex logic for brevity)

        # Recompile the model (Crucial step for Keras models after deserialization)
        self.model.compile(
            self.optimizer.optimizer,
            loss=self.optimizer.loss,
            metrics=self.optimizer.metrics,
        )

        # recover optimization result
        if state.get("_last_optimization_result") is ...:
            self._last_optimization_result = getattr(self.model, "history")

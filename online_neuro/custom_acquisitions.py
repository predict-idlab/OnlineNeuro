from abc import ABC, abstractmethod
from typing import Callable, Generic

import tensorflow as tf
from trieste.types import TensorType
from trieste.acquisition.sampler import ThompsonSampler
from trieste.acquisition.rule import AcquisitionRule, SearchSpace
from trieste.models.interfaces import ProbabilisticModel, ProbabilisticModelType
from trieste.acquisition.utils import select_nth_output
from typing import overload, Optional, Mapping
from trieste.types import Tag


class DiscreteMaxVarianceSampler(ThompsonSampler[ProbabilisticModel]):
    r"""
    This sampler selects points with the highest predictive variance over a given discrete set
    of input locations. This is useful for exploration-focused Bayesian optimization.
    """

    def sample(
            self,
            model: ProbabilisticModel,
            sample_size: int,
            at: TensorType,
            select_output: Callable[[TensorType], TensorType] = tf.reduce_mean,
    ) -> TensorType:
        """
        Samples points where the predictive variance is highest.

        :param model: The probabilistic model.
        :param sample_size: The number of points to sample.
        :param at: The candidate points to consider, with shape `[N, D]`.
        :param select_output: A method that extracts the relevant output from the model.
        :return: The sampled points, shape `[S, D]`.
        """
        tf.debugging.assert_positive(sample_size)
        tf.debugging.assert_shapes([(at, ["N", None])])

        mean, variance = model.predict(at)  # Compute mean and variance at given points
        selected_variance = select_output(variance, axis=-1)  # Select the relevant output
        # Select indices with highest variance
        indices = tf.argsort(selected_variance, direction="DESCENDING")
        indices = indices[:sample_size]
        return tf.gather(at, indices)  # Return the most uncertain points


class DiscreteMaxVarianceSampling(AcquisitionRule[TensorType, SearchSpace, ProbabilisticModelType]):
    r"""
    Implements Thompson sampling for choosing optimal points.

    This rule returns the minimizers of functions sampled from our model and evaluated across
    a discretization of the search space (containing `N` candidate points).

    The model is sampled either exactly (with an :math:`O(N^3)` complexity), or sampled
    approximately through a random Fourier `M` feature decompisition
    (with an :math:`O(\min(n^3,M^3))` complexity for a model trained on `n` points). The number
    `M` of Fourier features is specified when building the model.

    """

    @overload
    def __init__(
            self: "DiscreteMaxVarianceSampling[ProbabilisticModel]",
            num_search_space_samples: int,
            num_query_points: int,
            thompson_sampler: None = None,
            select_output: Callable[[TensorType], TensorType] = tf.reduce_mean,
    ):
        ...

    @overload
    def __init__(
            self: "DiscreteMaxVarianceSampling[ProbabilisticModelType]",
            num_search_space_samples: int,
            num_query_points: int,
            thompson_sampler: Optional[ThompsonSampler[ProbabilisticModelType]] = None,
            select_output: Callable[[TensorType], TensorType] = tf.reduce_mean,
    ):
        ...

    def __init__(
            self,
            num_search_space_samples: int,
            num_query_points: int,
            thompson_sampler: Optional[ThompsonSampler[ProbabilisticModelType]] = None,
            select_output: Callable[[TensorType], TensorType] = tf.reduce_mean,
    ):
        """
        :param num_search_space_samples: The number of points at which to sample the posterior.
        :param num_query_points: The number of points to acquire.
        :param thompson_sampler: Sampler to sample from the underlying model.
        :param select_output: A method that returns the desired trajectory from a trajectory
            sampler with shape [..., B], where B is a batch dimension. Defaults to the
            :func:~`trieste.acquisition.utils.select_nth_output` function with output dimension 0.
        """
        if not num_search_space_samples > 0:
            raise ValueError(f"Search space must be greater than 0, got {num_search_space_samples}")

        if not num_query_points > 0:
            raise ValueError(
                f"Number of query points must be greater than 0, got {num_query_points}"
            )

        if thompson_sampler is None:
            thompson_sampler = DiscreteMaxVarianceSampler(sample_min_value=False)

        self._thompson_sampler = thompson_sampler
        self._num_search_space_samples = num_search_space_samples
        self._num_query_points = num_query_points
        self._select_output = select_output

    def __repr__(self) -> str:
        """"""
        return f"""DiscreteMaxVarianceSampler(
            {self._num_search_space_samples!r},
            {self._num_query_points!r},
            {self._thompson_sampler!r},
            {self._select_output!r})"""

    def acquire(
            self,
            search_space: SearchSpace,
            models: Mapping[Tag, ProbabilisticModelType],
            datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> TensorType:
        """
        Sample `num_search_space_samples` (see :meth:`__init__`) points from the
        ``search_space``. Of those points, return the `num_query_points` points at which
        random samples yield the **minima** of the model posterior.

        :param search_space: The local acquisition search space for *this step*.
        :param models: The model of the known data. Uses the single key `OBJECTIVE`.
        :param datasets: The known observer query points and observations.
        :return: The ``num_query_points`` points to query.
        :raise ValueError: If ``models`` do not contain the key `OBJECTIVE`, or it contains any
            other key.
        """
        if models.keys() != {OBJECTIVE}:
            raise ValueError(
                f"dict of models must contain the single key {OBJECTIVE}, got keys {models.keys()}"
            )

        if datasets is None or datasets.keys() != {OBJECTIVE}:
            raise ValueError(
                f"""datasets must be provided and contain the single key {OBJECTIVE}"""
            )

        query_points = search_space.sample(self._num_search_space_samples)
        thompson_samples = self._thompson_sampler.sample(
            models[OBJECTIVE],
            self._num_query_points,
            query_points,
            select_output=self._select_output,
        )

        return thompson_samples
# /online_neuro/custom_acquisitions.py
import warnings
from abc import ABC, abstractmethod
from typing import Callable, Generic, Mapping, Optional, cast

import tensorflow as tf
import tensorflow_probability as tfp
from trieste.acquisition.interface import (
    AcquisitionFunction,
    AcquisitionFunctionClass,
    SingleModelAcquisitionBuilder,
)
from trieste.acquisition.rule import AcquisitionRule
from trieste.acquisition.utils import select_nth_output
from trieste.data import Dataset
from trieste.models.interfaces import ProbabilisticModel, ProbabilisticModelType
from trieste.observer import OBJECTIVE
from trieste.space import SearchSpace
from trieste.types import Tag, TensorType

from online_neuro.utils import CustomBox


class AbstractSampler(ABC, Generic[ProbabilisticModelType]):
    r"""
    A :class:`AbstractSampler` Similar to trieste.acquisition.sampler.ThompsonSampler.
    More general to allow for other sampling methods and not just minimization by Thompson.
    Currently we implement a Minimization of Variance, but other Integrate or variations can be used.
    """

    def __init__(self):
        """
        Nothing is implemented here.
        This can be used for internal logic such as ._sample_min_value case in Trieste.

        """

    def __repr__(self) -> str:
        """"""
        return f"""{self.__class__.__name__}
        """

    @abstractmethod
    def sample(
        self,
        model: ProbabilisticModelType,
        sample_size: int,
        at: TensorType,
        select_output: Callable[[TensorType], TensorType] = select_nth_output,
    ) -> TensorType:
        """
        :param model: The model to sample from.
        :param sample_size: The desired number of samples.
        :param at: Input points that define the sampler.
        :param select_output: A method that returns the desired output from the model sampler, with
            shape `[S, N]` where `S` is the number of samples and `N` is the number of locations.
            Defaults to the :func:~`trieste.acquisition.utils.select_nth_output` function with
            output dimension 0.
        :return: Samples.
        """


class ExpectedImprovementXsi(SingleModelAcquisitionBuilder[ProbabilisticModel]):
    """
    Builder for the expected improvement function where the "best" value is taken to be the minimum
    of the posterior mean at observed points. The trade-off between exploitation and exploration
    can be controlled with the `xsi` parameter.
    As described in https://ekamperi.github.io/machine%20learning/2021/06/11/acquisition-functions.html
    Probably first mentioned in "Efficient Global Optimization of Expensive Black-Box Functions" by Jones, Schonlau, and Welch (1998).
    """

    def __init__(self, search_space: Optional[SearchSpace] = None, xsi: float = 0.01):
        """
        :param search_space: The global search space over which the optimisation is defined.
        :param xsi: The hyperparameter to control the trade-off between exploitation and
            exploration. Higher values lead to more exploration.
        """
        self._search_space = search_space
        self._xsi = xsi

    def __repr__(self) -> str:
        """"""
        return f'ExpectedImprovement({self._search_space!r}, xsi={self._xsi!r})'

    def prepare_acquisition_function(
        self,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param model: The model.
        :param dataset: The data from the observer. Must be populated.
        :return: The expected improvement function.
        """
        tf.debugging.Assert(dataset is not None, [tf.constant([])])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message='Dataset must be populated.')

        if self._search_space is not None and self._search_space.has_constraints:
            is_feasible = self._search_space.is_feasible(dataset.query_points)
            if not tf.reduce_any(is_feasible):
                query_points = dataset.query_points
            else:
                query_points = tf.boolean_mask(dataset.query_points, is_feasible)
        else:
            is_feasible = tf.constant([True], dtype=bool)
            query_points = dataset.query_points

        mean, _ = model.predict(query_points)
        if not tf.reduce_any(is_feasible):
            eta = tf.reduce_max(mean, axis=0)
        else:
            eta = tf.reduce_min(mean, axis=0)

        return expected_improvement_xsi(model, eta, self._xsi)

    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param function: The acquisition function to update.
        :param model: The model.
        :param dataset: The data from the observer. Must be populated.
        """
        tf.debugging.Assert(dataset is not None, [tf.constant([])])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message='Dataset must be populated.')
        tf.debugging.Assert(isinstance(function, expected_improvement_xsi), [tf.constant([])])

        if self._search_space is not None and self._search_space.has_constraints:
            is_feasible = self._search_space.is_feasible(dataset.query_points)
            if not tf.reduce_any(is_feasible):
                query_points = dataset.query_points
            else:
                query_points = tf.boolean_mask(dataset.query_points, is_feasible)
        else:
            is_feasible = tf.constant([True], dtype=bool)
            query_points = dataset.query_points

        mean, _ = model.predict(query_points)
        if not tf.reduce_any(is_feasible):
            eta = tf.reduce_max(mean, axis=0)
        else:
            eta = tf.reduce_min(mean, axis=0)

        function.update(eta)  # type: ignore
        return function


class expected_improvement_xsi(AcquisitionFunctionClass):
    def __init__(self, model: ProbabilisticModel, eta: TensorType, xsi: float = 0.01):
        r"""
        Return the Expected Improvement (EI) acquisition function.
        :param model: The model of the objective function.
        :param eta: The "best" observation.
        :param xsi: The hyperparameter to control the trade-off between exploitation and
            exploration. if > 0, the algorithm is biased towards exploration. if < 0, it is
            biased towards exploitation.
        """
        self._model = model
        self._eta = tf.Variable(eta)
        self._xsi = xsi

    def update(self, eta: TensorType) -> None:
        """Update the acquisition function with a new eta value."""
        self._eta.assign(eta)

    @tf.function
    def __call__(self, x: TensorType) -> TensorType:
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message='This acquisition function only supports batch sizes of one.',
        )
        mean, variance = self._model.predict(tf.squeeze(x, -2))
        normal = tfp.distributions.Normal(mean, tf.sqrt(variance))
        ei = (self._eta - mean - self._xsi) * normal.cdf(self._eta - self._xsi) + variance * normal.prob(self._eta - self._xsi)
        # ei = (self._eta - mean - self._xsi) * normal.cdf(self._eta - self._xsi) + variance * normal.prob(eta_shifted)
        # augmentation = 1 - (tf.math.sqrt(self._noise_variance)) / (
        #     tf.math.sqrt(self._noise_variance + variance)
        # )
        return ei


class TemporalVariance(AbstractSampler[ProbabilisticModel]):
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
        tf.debugging.assert_shapes([(at, ['N', None])])

        mean, variance = model.predict(at)  # Compute mean and variance at given points
        selected_variance = select_output(variance, axis=-1)  # Select the relevant output
        indices = tf.argsort(selected_variance, direction='DESCENDING')
        indices = indices[:sample_size]
        return tf.gather(at, indices)  # Return the most uncertain points

    def __repr__(self) -> str:
        """"""
        return f"""{self.__class__.__name__}
        """


class TemporalVarianceMovement(AbstractSampler[ProbabilisticModel]):
    r"""
    TODO: Finish this part to account for "movement" (first derivative)
    This sampler selects points with the highest predictive variance over a given discrete set
    of input locations. This is useful for exploration-focused Bayesian optimization.
    """
    def __init__(self, alpha: float = 0.5):
        warnings.warn('TemporalVarianceMovement is not fully implemented yet.')
        self.alpha = alpha

    def sample(
            self,
            model: ProbabilisticModel,
            sample_size: int,
            at: TensorType,
            select_output: Callable[[TensorType], TensorType] = tf.reduce_mean
    ) -> TensorType:
        """
        Samples points where the predictive variance is highest.

        :param model: The probabilistic model.
        :param sample_size: The number of points to sample.
        :param at: The candidate points to consider, with shape `[N, D]`.
        :param select_output: A method that extracts the relevant output from the model.
        :return: The sampled points, shape `[S, D]`.
        """
        # TODO completet this method if still needed
        tf.debugging.assert_positive(sample_size)
        tf.debugging.assert_shapes([(at, ['N', None])])

        mean, variance = model.predict(at)  # Compute mean and variance at given points
        # movement = tf.concat([
        #     tf.zeros_like(mean[:1]),  # pad the first value
        #     tf.abs(mean[1:] - mean[:-1])
        # ], axis=0)

        score = self.alpha * mean + (1 - self.alpha) * variance
        selected_score = select_output(score, axis=-1)  # Select the relevant output
        # Select indices with the highest variance
        indices = tf.argsort(selected_score, direction='DESCENDING')
        indices = indices[:sample_size]
        return tf.gather(at, indices)  # Return the most uncertain points

    def __repr__(self) -> str:
        """"""
        return f"""{self.__class__.__name__}, alpha ={self.alpha}

        """


class DiscreteMaxVarianceSampling(AcquisitionRule[TensorType, SearchSpace, ProbabilisticModelType]):
    r"""
    Implements sampling of candidates to propose a highest uncertainty point

    """

    def __init__(
            self: 'DiscreteMaxVarianceSampling[ProbabilisticModel]',
            sample_size: int,
            query_points: int,
            sampler: Optional[AbstractSampler] = None,
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
        if not sample_size > 0:
            raise ValueError(f'Search space must be greater than 0, got {sample_size}')

        if not query_points > 0:
            raise ValueError(
                f'Number of query points must be greater than 0, got {query_points}'
            )

        if sampler is None:
            sampler = TemporalVariance()

        self._sampler = sampler
        self._sample_size = sample_size
        self._query_points = query_points
        self._select_output = select_output

    def __repr__(self) -> str:
        """"""
        return f"""DiscreteMaxVarianceSampler(
            {self._sample_size!r},
            {self._query_points!r},
            {self._sampler!r},
            {self._select_output!r})"""

    def acquire(
            self,
            search_space: SearchSpace,
            models: Mapping[Tag, ProbabilisticModelType],
            # datasets: Optional[Mapping[Tag, Dataset]] = None,
            datasets: Optional[Mapping] = None,
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
        assert isinstance(search_space, CustomBox)

        if models.keys() != {OBJECTIVE}:
            raise ValueError(
                f'dict of models must contain the single key {OBJECTIVE}, got keys {models.keys()}'
            )

        if datasets is None or datasets.keys() != {OBJECTIVE}:
            raise ValueError(
                f"""datasets must be provided and contain the single key {OBJECTIVE}"""
            )

        query_points = search_space.sample_method(self._sample_size, sampling_method='random')
        samples = self._sampler.sample(
            models[OBJECTIVE],
            self._query_points,
            query_points,
            select_output=self._select_output,
        )

        return samples

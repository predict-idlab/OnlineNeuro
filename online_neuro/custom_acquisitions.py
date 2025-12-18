# /online_neuro/custom_acquisitions.py
import warnings
from abc import ABC, abstractmethod
from typing import Callable, Generic, Mapping, Optional, cast

import tensorflow as tf
import tensorflow_probability as tfp
from trieste.acquisition.function.active_learning import (
    BayesianActiveLearningByDisagreement,
    PredictiveVariance,
)
from trieste.acquisition.function.function import NegativePredictiveMean
from trieste.acquisition.function.multi_objective import ExpectedHypervolumeImprovement
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

from online_neuro.utils.constants import ProblemType
from online_neuro.utils.custom_box import CustomBox
from online_neuro.utils.type_mappers import get_problem_class


class AbstractSampler(ABC, Generic[ProbabilisticModelType]):
    """
    Abstract base class for custom sampling methods used to select query points
    based on properties of the probabilistic model (e.g., mean, variance, or custom scores).

    This serves a similar purpose to Trieste's internal samplers but provides
    a more general interface for non-Thompson sampling methods, allowing for
    strategies focused purely on exploration (like minimizing variance).
    """

    def __init__(self):
        """
        Nothing is implemented here.
        This can be used for internal logic such as ._sample_min_value case in Trieste.

        """
        pass

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
        Generates query points by analyzing the model's prediction properties
        at a given set of candidate locations.

        Parameters
        ----------
        model : ProbabilisticModelType
            The probabilistic model to sample from (e.g., a GP).
        sample_size : int
            The desired number of points to select from the candidate set `at`.
        at : TensorType
            Candidate input points (locations) with shape `[N, D]`.
        select_output : Callable[[TensorType], TensorType], optional
            A method that selects the relevant output dimension from the model's
            prediction (e.g., for multi-output models). Defaults to
            :func:`~trieste.acquisition.utils.select_nth_output` (output 0).

        Returns
        -------
        TensorType
            The selected sample points, shape `[sample_size, D]`.
        """


class ExpectedImprovementXsi(SingleModelAcquisitionBuilder[ProbabilisticModel]):
    """
    Builder for the Expected Improvement (EI) function incorporating the ξ
    trade-off parameter (also denoted ξ).

    The "best" value (η) is calculated as the minimum posterior mean at
    observed feasible points. The `xsi` parameter controls the balance between
    exploitation (minimizing the mean) and exploration.

    As described in https://ekamperi.github.io/machine%20learning/2021/06/11/acquisition-functions.html
    Probably first mentioned in 'Efficient Global Optimization of Expensive Black-Box Functions'
    by Jones, Schonlau, and Welch (1998).
    """

    def __init__(self, search_space: SearchSpace | None = None, xsi: float = 0.01):
        """
        Initializes the EI builder.

        Parameters
        ----------
        search_space : SearchSpace, optional
            The global search space. Used to check for feasible points when
            determining the current best observation ($\eta$).
        xsi : float, optional
            The hyperparameter controlling the trade-off. Higher values (xsi > 0)
            encourage more exploration. Defaults to 0.01.
        """
        self._search_space = search_space
        self._xsi = xsi

    def __repr__(self) -> str:
        """Returns a string representation including search space and xsi."""
        return f"ExpectedImprovement({self._search_space!r}, xsi={self._xsi!r})"

    def prepare_acquisition_function(
        self,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        Calculates the current best value ($\eta$) and returns an initialized
        `expected_improvement_xsi` acquisition function instance.
        """
        tf.debugging.Assert(dataset is not None, [tf.constant([])])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")

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
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        tf.debugging.Assert(
            isinstance(function, expected_improvement_xsi), [tf.constant([])]
        )

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

        self._dtype = self._eta.dtype
        self._xsi = tf.constant(xsi, dtype=self._dtype)

    def update(self, eta: TensorType) -> None:
        """Update the acquisition function with a new eta value."""
        self._eta.assign(eta)

    @tf.function
    def __call__(self, x: TensorType) -> TensorType:
        """
        Calculates the Expected Improvement at candidate points `x`.

        Parameters
        ----------
        x : TensorType
            Candidate query points, expected shape `[..., 1, D]`.

        Returns
        -------
        TensorType
            The EI values for each candidate point.
        """
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This acquisition function only supports batch sizes of one.",
        )
        # Model prediction uses the input dtype (x.dtype)
        mean, variance = self._model.predict(tf.squeeze(x, -2))

        sigma = tf.sqrt(variance)

        # Ensure all intermediate operations use the same high precision dtype:

        # Calculate Improvement term I = eta - mean - xsi
        improvement = self._eta - mean - self._xsi

        # Numerical tolerance for zero variance (must match dtype)
        epsilon = tf.constant(1e-12, dtype=self._dtype)
        is_nonzero_sigma = sigma > epsilon

        # Initialize Standard Normal distribution (must match dtype)
        standard_normal = tfp.distributions.Normal(
            loc=tf.zeros_like(mean), scale=tf.ones_like(sigma), allow_nan_stats=False
        )

        # --- Case 1: Non-zero variance (Standard EI formula) ---

        # Z = I / sigma
        # Use tf.math.divide_no_nan for safety, although tf.where handles the zero case later
        Z = tf.where(is_nonzero_sigma, improvement / sigma, tf.zeros_like(improvement))

        # EI = I * Phi(Z) + sigma * phi(Z)
        ei_nonzero = improvement * standard_normal.cdf(
            Z
        ) + sigma * standard_normal.prob(Z)

        # --- Case 2: Zero variance (Deterministic Improvement) ---
        # EI = max(0, I)
        ei_zero = tf.maximum(tf.constant(0.0, dtype=self._dtype), improvement)

        # Combine results
        ei = tf.where(is_nonzero_sigma, ei_nonzero, ei_zero)

        return ei


class TemporalVariance(AbstractSampler[ProbabilisticModel]):
    """
    A custom sampler that selects query points corresponding to the highest
    predictive variance across a set of candidates.

    This implements a purely exploration-driven sampling strategy, maximizing
    uncertainty to gather information efficiently.
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

        Parameters
        ----------
        model : ProbabilisticModel
            The probabilistic model.
        sample_size : int
            The number of points to sample.
        at : TensorType
            The candidate points to consider, with shape `[N, D]`.
        select_output : Callable[[TensorType], TensorType], optional
            A method that aggregates or selects the relevant variance output
            if the variance prediction is multi-dimensional.

        Returns
        -------
        TensorType
            The sampled points, shape `[sample_size, D]`.
        """
        tf.debugging.assert_positive(sample_size)
        tf.debugging.assert_shapes([(at, ["N", None])])

        mean, variance = model.predict(at)  # Compute mean and variance at given points
        selected_variance = select_output(variance, axis=-1)

        # Select indices corresponding to the highest variance
        indices = tf.argsort(selected_variance, direction="DESCENDING")
        indices = indices[:sample_size]
        return tf.gather(at, indices)  # Return the most uncertain points

    def __repr__(self) -> str:
        """"""
        return f"""{self.__class__.__name__}
        """


class TemporalVarianceMovement(AbstractSampler[ProbabilisticModel]):
    """
    A sampler intended to select points based on a weighted combination
    of predictive mean and predictive variance ("movement").

    Score(x) = \alpha * |Gradient(Mean(x))| + (1 - \alpha) * Variance(x)
    """

    def __init__(self, alpha: float = 0.5):
        """
        Initializes the sampler.

        Parameters
        ----------
        alpha : float, optional
            Weighting factor controlling the balance between mean (exploitation)
            and variance (exploration). A higher alpha weights the mean more heavily.
            Defaults to 0.5.
        """
        if not 0 <= alpha <= 1:
            raise ValueError("Alpha must be between 0.0 and 1.0.")

        self.alpha = tf.constant(alpha, dtype=tf.float32)

    def _normalize_tensor(self, t: tf.Tensor) -> tf.Tensor:
        """Performs Min-Max scaling to map tensor values to [0, 1]."""
        t = tf.squeeze(t)
        t_min = tf.reduce_min(t)
        t_max = tf.reduce_max(t)

        # Handle the case where max == min (zero range)
        range_ = tf.maximum(t_max - t_min, tf.constant(1e-9, dtype=t.dtype))
        output = (t - t_min) / range_

        return output

    def sample(
        self,
        model: ProbabilisticModel,
        sample_size: int,
        at: TensorType,
        select_output: Callable[[TensorType], TensorType] = tf.reduce_mean,
    ) -> TensorType:
        """
        Samples points based on the combined score of local sensitivity and uncertainty,
        using normalized components.
        """
        tf.debugging.assert_positive(sample_size)
        tf.debugging.assert_shapes([(at, ["N", None])])

        # Ensure 'at' is watchable by the GradientTape
        at = tf.cast(at, self.alpha.dtype)
        at_watched = tf.Variable(at, trainable=True)

        # --- 1. Compute Mean, Variance, and Gradient ---
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(at_watched)
            mean, variance = model.predict(at_watched)

            selected_mean = select_output(mean, axis=-1)
            selected_variance = select_output(variance, axis=-1)

        mean_gradient = tape.gradient(selected_mean, at_watched)

        # --- 2. Calculate Raw Sensitivity and Uncertainty Scores ---

        if mean_gradient is None:
            warnings.warn("Gradient computation failed, falling back to pure variance.")
            raw_sensitivity_score = tf.zeros_like(selected_variance)
        else:
            raw_sensitivity_score = tf.norm(mean_gradient, axis=-1)

        raw_uncertainty_score = selected_variance

        # --- 3. Normalization (Min-Max Scaling) ---

        # If there's only one candidate point, scaling is trivial (or unnecessary)
        if tf.shape(at)[0] > 1:
            sensitivity_norm = self._normalize_tensor(raw_sensitivity_score)
            uncertainty_norm = self._normalize_tensor(raw_uncertainty_score)
        else:
            # Avoid scaling single points
            sensitivity_norm = raw_sensitivity_score
            uncertainty_norm = raw_uncertainty_score

        # --- 4. Calculate Combined Score ---
        alpha_tensor = tf.cast(self.alpha, sensitivity_norm.dtype)

        score = (alpha_tensor * sensitivity_norm) + (
            (tf.constant(1.0, dtype=alpha_tensor.dtype) - alpha_tensor)
            * uncertainty_norm
        )

        # 5. Select and Return
        indices = tf.argsort(score, direction="DESCENDING")
        indices = indices[:sample_size]

        return tf.gather(at, indices)

    def __repr__(self) -> str:
        """ """
        return f"{self.__class__.__name__}, alpha ={self.alpha}"


class DiscreteMaxVarianceSampling(
    AcquisitionRule[TensorType, SearchSpace, ProbabilisticModelType]
):
    """
    An Acquisition Rule that implements batch sampling by selecting the top `K`
    points with the highest uncertainty/variance from a larger set of candidates
    randomly sampled across the search space.

    This provides a discrete, high-exploration alternative to standard analytical
    acquisition function optimization (like EGO).
    """

    def __init__(
        self: "DiscreteMaxVarianceSampling[ProbabilisticModel]",
        sample_size: int,
        query_points: int,
        sampler: Optional[AbstractSampler] = None,
        select_output: Callable[[TensorType], TensorType] = tf.reduce_mean,
    ):
        """
        Initializes the Discrete Max Variance Sampling rule.

        Parameters
        ----------
        sample_size : int
            The number of random candidate points to sample from the search space
            at each acquisition step (N in N-out-of-M selection). Must be > 0.
        query_points : int
            The number of final query points to select from the candidates (K in N-out-of-M).
            Must be > 0.
        sampler : Optional[AbstractSampler], optional
            The specific sampling strategy to use for ranking candidates.
            Defaults to :class:`TemporalVariance`.
        select_output : Callable[[TensorType], TensorType], optional
            Method to select the relevant output dimension for multi-output models.
            Defaults to `tf.reduce_mean`.
        """
        if not sample_size > 0:
            raise ValueError(f"Search space must be greater than 0, got {sample_size}")

        if not query_points > 0:
            raise ValueError(
                f"Number of query points must be greater than 0, got {query_points}"
            )

        if sampler is None:
            sampler = TemporalVariance()

        self._sampler = sampler
        self._sample_size = sample_size
        self._query_points = query_points
        self._select_output = select_output

    def __repr__(self) -> str:
        """Return a detailed string representation of the rule."""
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
        Executes the acquisition step: randomly sample candidates, use the internal
        sampler to rank them by uncertainty, and select the top `query_points`.

        Parameters
        ----------
        search_space : SearchSpace
            The local acquisition search space for *this step*. Must be an instance
            of `CustomBox`.
        models : Mapping[Tag, ProbabilisticModelType]
            The model of the known data. Must contain the single key `OBJECTIVE`.
        datasets : Optional[Mapping[Tag, Dataset]], optional
            The known observer query points and observations. Must contain the
            single key `OBJECTIVE`.

        Returns
        -------
        TensorType
            The selected query points (batch size: `self._query_points`).

        Raises
        ------
        ValueError
            If the required model or dataset keys (`OBJECTIVE`) are missing.
        AssertionError
            If `search_space` is not `CustomBox`.
        """
        assert isinstance(search_space, CustomBox)

        if models.keys() != {OBJECTIVE}:
            raise ValueError(
                f"dict of models must contain the single key {OBJECTIVE}, got keys {models.keys()}"
            )

        if datasets is None or datasets.keys() != {OBJECTIVE}:
            raise ValueError(
                f"""datasets must be provided and contain the single key {OBJECTIVE}"""
            )
        # 1. Sample candidates randomly from the search space
        query_points = search_space.sample_method(
            self._sample_size, sampling_method="random"
        )

        # 2. Use the sampler to rank and select the best K points
        samples = self._sampler.sample(
            models[OBJECTIVE],
            self._query_points,
            query_points,
            select_output=self._select_output,
        )

        return samples


def select_acquisition_function(
    problem_type: str, acq_name: str, verbose: bool = True
) -> SingleModelAcquisitionBuilder:
    """
    Return an appropriate  acquisition function based on the problem type
    and optional acquisition name.

    The logic is:

    - **Multiobjective** → ``ExpectedHypervolumeImprovement``
    - **Classification** → ``BayesianActiveLearningByDisagreement``
    - **Regression**:
        - ``"negative_predictive_mean"`` → ``NegativePredictiveMean``
        - ``"predictive_variance"`` → ``PredictiveVariance``
        - Any other name → default to ``PredictiveVariance``

    Parameters
    ----------
    problem_type : str
        High-level problem type (e.g., ``"regression"``, ``"multiobjective"``,
        ``"classification"``). Case-insensitive.

    acq_name : str
        Name of the acquisition function to use for regression problems.
        Ignored for classification and multiobjective problems.

    verbose : bool, optional
        If True, print which acquisition function was selected.

    Returns
    -------
    Any
        An instantiated Trieste(or other) acquisition function.

    Raises
    ------
    NotImplementedError
        If no implementation exists for the given problem type.

    TODO
    ----
    # Extend and implement batch sampling acquisitions

    Extend acquisitions as required. Currently Trieste includes the following:
    # AugmentedExpectedImprovement
    # ExpectedImprovement
    # ProbabilityOfImprovement
    # NegativeLowerConfidenceBound
    # NegativePredictiveMean
    # ProbabilityOfFeasibility
    # FastConstraintsFeasibility
    # ExpectedConstrainedImprovement
    # MonteCarloExpectedImprovement
    # MonteCarloAugmentedExpectedImprovement
    # MonteCarloExpectedImprovement
    # BatchMonteCarloExpectedImprovement
    # BatchExpectedImprovement
    # MultipleOptimismNegativeLowerConfidenceBound
    """
    problem_class = get_problem_class(problem_type)

    acq_map = {
        ProblemType.MULTIOBJECTIVE: ExpectedHypervolumeImprovement,
        ProblemType.CLASSIFICATION: BayesianActiveLearningByDisagreement,
    }
    # Currently there's only one implementation for classification and one for MOO.
    if problem_class in acq_map:
        msg = f"Using acquisition function for {problem_class.name}"
        acq = acq_map[problem_class]()

    # Handle Regression case with more options
    elif problem_class == ProblemType.REGRESSION:
        if acq_name == "negative_predictive_mean":
            msg = "Using Negative PredictiveMean for regression."
            acq = NegativePredictiveMean()
        elif acq_name == "predictive_variance":
            msg = "Using PredictiveVariance for regression."
            acq = PredictiveVariance()
        else:
            warnings.warn(
                f"Acquisition '{acq_name}' not identified for regression. Defaulting to Predictive Variance."
            )
            msg = "Using PredictiveVariance for regression."
            acq = PredictiveVariance()
    else:
        raise NotImplementedError(
            f"No implementation for problem of class {problem_class}"
        )

    if verbose:
        print(msg)

    return acq

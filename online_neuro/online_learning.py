from trieste.models.gpflow.builders import build_vgp_classifier, build_gpr, build_svgp, build_sgpr
from trieste.models.gpflow import (SparseVariational, VariationalGaussianProcess,
                                   SparseGaussianProcessRegression, GaussianProcessRegression)
from trieste.models.gpflow import KMeansInducingPointSelector
from trieste.models.optimizer import BatchOptimizer

import tensorflow as tf
import tensorflow_probability as tfp

from functools import partial
from sklearn.preprocessing import StandardScaler
import math
import warnings
import matplotlib.pyplot as plt
import gpflow
from gpflow.models import GPR, SGPR, SVGP, VGP, GPModel
from trieste.data import Dataset
from trieste.space import Box, SearchSpace
from typing import Optional, Sequence, Callable, Hashable, Tuple, TypeVar, Union

TensorType = Union[tf.Tensor, tf.Variable]

KERNEL_PRIOR_SCALE = tf.constant(1.0, dtype=gpflow.default_float())

DEFAULT_KERNEL_VARIANCE = tf.constant(1, dtype=gpflow.default_float())
"""
Default kernel variance for regression.
"""

DEFAULT_KERNEL_VARIANCE_NOISE_FREE = tf.constant(100.0, dtype=gpflow.default_float())
"""
Default kernel variance for noise free regression.
"""

MAX_NUM_INDUCING_POINTS = tf.constant(200, dtype=tf.int32)
"""
Default maximum number of inducing points.
"""

NUM_INDUCING_POINTS_PER_DIM = tf.constant(15, dtype=tf.int32)
"""
Default number of inducing points per dimension of the search space.
"""

SIGNAL_NOISE_RATIO_LIKELIHOOD = tf.constant(10, dtype=gpflow.default_float())
"""
Default value used for initializing (noise) variance parameter of the likelihood function.
If user does not specify it, the noise variance is set to maintain the signal to noise ratio
determined by this default value. Signal variance in the kernel is set to the empirical variance.
"""


def _set_gaussian_likelihood_variance(
    model: GPModel, variance: TensorType, likelihood_variance: Optional[float]
) -> None:
    if likelihood_variance is None:
        noise_variance = variance / SIGNAL_NOISE_RATIO_LIKELIHOOD**2
    else:
        tf.debugging.assert_positive(likelihood_variance)
        noise_variance = tf.cast(likelihood_variance, dtype=gpflow.default_float())

    model.likelihood.variance = gpflow.base.Parameter(
        noise_variance, transform=gpflow.utilities.positive(lower=1e-12)
    )


def _get_inducing_points(search_space: SearchSpace,
                         num_inducing_points: Optional[int]) -> TensorType:
    if num_inducing_points is not None:
        tf.debugging.assert_positive(num_inducing_points)
    else:
        num_inducing_points = min(MAX_NUM_INDUCING_POINTS,
                                  NUM_INDUCING_POINTS_PER_DIM * search_space.dimension)
    if isinstance(search_space, Box):
        inducing_points = search_space.sample_sobol(num_inducing_points)
    else:
        inducing_points = search_space.sample(num_inducing_points)
    return inducing_points


def _get_data_stats(data: Dataset) -> tuple[TensorType, TensorType, int]:
    empirical_variance = tf.math.reduce_variance(data.observations)
    empirical_mean = tf.math.reduce_mean(data.observations)
    num_data_points = len(data.observations)

    return empirical_mean, empirical_variance, num_data_points


def _get_mean_function(mean: TensorType) -> gpflow.mean_functions.MeanFunction:
    mean_function = gpflow.mean_functions.Constant(mean)
    return mean_function


def _get_lengthscales(search_space: SearchSpace, default_lengthscale=2) -> TensorType:
    lengthscales = (default_lengthscale * (search_space.upper - search_space.lower) * math.sqrt(search_space.dimension))
    search_space_collapsed = tf.equal(search_space.upper, search_space.lower)
    lengthscales = tf.where(
        search_space_collapsed, tf.constant(1.0, dtype=gpflow.default_float()), lengthscales
    )
    return lengthscales


def _get_kernel(
    variance: TensorType,
    search_space: SearchSpace,
    add_prior_to_lengthscale: bool,
    add_prior_to_variance: bool,
) -> gpflow.kernels.Kernel:
    lengthscales = _get_lengthscales(search_space)

    kernel = gpflow.kernels.Matern52(variance=variance, lengthscales=lengthscales)

    if add_prior_to_lengthscale:
        kernel.lengthscales.prior = tfp.distributions.LogNormal(
            tf.math.log(lengthscales), KERNEL_PRIOR_SCALE
        )
    if add_prior_to_variance:
        kernel.variance.prior = tfp.distributions.LogNormal(
            tf.math.log(variance), KERNEL_PRIOR_SCALE
        )

    return kernel


def build_vgp(data: Dataset,
              search_space: SearchSpace,
              kernel_priors: bool = True,
              likelihood_variance : Optional[float] = None,
              trainable_likelihood: bool = False,
              ) -> VGP:
    """
    Build a :class:`~gpflow.models.VGP` binary classification model with sensible initial
    parameters and priors. We use :class:`~gpflow.kernels.Matern52` kernel and
    :class:`~gpflow.mean_functions.Constant` mean function in the model. We found the default
    configuration used here to work well in most situation, but it should not be taken as a
    universally good solution.

    We set priors for kernel hyperparameters by default in order to stabilize model fitting. We
    found the priors below to be highly effective for objective functions defined over the unit
    hypercube. They do seem to work for other search space sizes, but we advise caution when using
    them in such search spaces. Using priors allows for using maximum a posteriori estimate of
    these kernel parameters during model fitting. In the ``noise_free`` case we do not use prior
    for the kernel variance parameters.

    Note that although we scale parameters as a function of the size of the search space, ideally
    inputs should be normalised to the unit hypercube before building a model.

    :param data: Dataset from the initial design, used for estimating the variance of observations.
    :param search_space: Search space for performing Bayesian optimization, used for scaling the
        parameters.
    :param kernel_priors: If set to `True` (default) priors are set for kernel parameters (variance
        and lengthscale). In the ``noise_free`` case kernel variance prior is not set.
    :param likelihood_variance: Likelihood (noise) variance parameter can be optionally set to a
        certain value. If left unspecified (default), the noise variance is set to maintain the
        signal to noise ratio of value given by ``SIGNAL_NOISE_RATIO_LIKELIHOOD``, where signal
        variance in the kernel is set to the empirical variance. This argument is ignored in the
        classification case.
    :param trainable_likelihood: If set to `True` likelihood parameter is set to
        be trainable. By default set to `False`.
    :return: A :class:`~gpflow.models.VGP` model.
    """
    empirical_mean, empirical_variance, num_data_points = _get_data_stats(data)

    model_likelihood = gpflow.likelihoods.Gaussian()

    kernel = _get_kernel(empirical_variance, search_space, kernel_priors, kernel_priors)
    mean = _get_mean_function(empirical_mean)

    model = VGP(data.astuple(), kernel, model_likelihood, mean_function=mean)

    _set_gaussian_likelihood_variance(model, empirical_variance, likelihood_variance)
    gpflow.set_trainable(model.likelihood, trainable_likelihood)

    gpflow.set_trainable(model.kernel.variance, True)

    return model


def build_model(init_dataset, search_space, config, **kwargs):
    """
    @param init_dataset:
    @param search_space:
    @param config:
    @return:
    """
    if config['classification']:
        if config['variational'] and config['sparse']:
            # TODO
            if config['noise_free']:
                msg = "build_svgp by Trieste does not have a noisy version, need to rewrite this function(TODO)"
                warnings.warn(msg)
            gpflow_model = build_svgp(init_dataset, search_space, classification=True,
                                      #noise_free=config['experiment']['noise_free'],
                                      trainable_likelihood=config['trainable_likelihood'],
                                      )
            model = VariationalGaussianProcess(gpflow_model)
        elif config['variational'] and ~config['sparse']:
            gpflow_model = build_vgp_classifier(init_dataset, search_space,
                                                noise_free=config['noise_free']
                                                )

            model = VariationalGaussianProcess(gpflow_model)
        else:
            raise Exception("Classification not implemented with non variational GPs")
    else:
        if config['variational'] and config['sparse']:
            gpflow_model = build_svgp(init_dataset, search_space,
                                      classification=False,
                                      likelihood_variance=config['kernel_variance'],
                                      trainable_likelihood=config['trainable_likelihood'],
                                      num_inducing_points=20
                                      )
            inducing_point_selector = KMeansInducingPointSelector()

            model = SparseVariational(
                gpflow_model,
                num_rff_features=1000,
                inducing_point_selector=inducing_point_selector,
                likelihood_variance=config['kernel_variance'],
                optimizer=BatchOptimizer(tf.optimizers.Adam(0.1), max_iter=100,
                                         batch_size=50, compile=True),
            )
        elif config['variational'] and ~config['sparse']:
            gpflow_model = build_vgp(init_dataset, search_space,
                                     likelihood_variance=config['kernel_variance'],
                                     trainable_likelihood=config['trainable_likelihood'],
                                     )

            model = VariationalGaussianProcess(gpflow_model)
        elif ~config['variational'] and config['sparse']:
            gpflow_model = build_sgpr(init_dataset, search_space,
                                      likelihood_variance=config['kernel_variance'],
                                      trainable_likelihood=config['trainable_likelihood'],
                                      )
            inducing_point_selector = KMeansInducingPointSelector()

            model = SparseGaussianProcessRegression(
                gpflow_model,
                num_rff_features=1000,
                inducing_point_selector=inducing_point_selector,
                optimizer=BatchOptimizer(tf.optimizers.Adam(0.1), max_iter=100,
                                         batch_size=50, compile=True)
            )
        elif ~config['variational'] and ~config['sparse']:
            gpflow_model = build_gpr(init_dataset, search_space,
                                     likelihood_variance=config['kernel_variance'],
                                     trainable_likelihood=config['trainable_likelihood'],
                                     )

            model = GaussianProcessRegression(gpflow_model)

        else:
            raise Exception("Classification not implemented with non variational GPs")
    return model

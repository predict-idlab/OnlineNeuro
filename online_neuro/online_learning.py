# /online_neuro/online_learning.py

# Notice that partial code is based on The Trieste repository
# https://github.com/secondmind-labs/trieste
import warnings
from typing import Union

import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.models import VGP, GPModel
from trieste.data import Dataset
from trieste.models.gpflow.builders import (
    _get_data_stats,
    _get_lengthscales,
    _get_mean_function,
    build_gpr,
    build_sgpr,
    build_svgp,
    build_vgp_classifier,
)
from trieste.models.gpflow.inducing_point_selectors import KMeansInducingPointSelector
from trieste.models.gpflow.models import (
    GaussianProcessRegression,
    SparseGaussianProcessRegression,
    SparseVariational,
    VariationalGaussianProcess,
)
from trieste.models.optimizer import BatchOptimizer
from trieste.space import SearchSpace

TensorType = Union[tf.Tensor, tf.Variable]

# Default constants as per Trieste
#   On the long term also expose some of these parameters for specific cases.
KERNEL_PRIOR_SCALE = tf.constant(1.0, dtype=gpflow.default_float())
DEFAULT_KERNEL_VARIANCE = tf.constant(1.0, dtype=gpflow.default_float())
DEFAULT_KERNEL_VARIANCE_NOISE_FREE = tf.constant(100.0, dtype=gpflow.default_float())
MAX_NUM_INDUCING_POINTS = tf.constant(200, dtype=tf.int32)
NUM_INDUCING_POINTS_PER_DIM = tf.constant(15, dtype=tf.int32)
SIGNAL_NOISE_RATIO_LIKELIHOOD = tf.constant(10, dtype=gpflow.default_float())


## Trieste functions that use default values internally (here for easier access and maintenance)
def _set_gaussian_likelihood_variance(
    model: GPModel, variance: TensorType, likelihood_variance: float | None = None
) -> None:
    """
    Set the Gaussian likelihood variance for a GP model.

    Parameters
    ----------
    model : GPModel
        The GPflow model whose likelihood variance is being set.
    variance : TensorType
        Empirical variance of the dataset to scale the likelihood variance if not provided.
    likelihood_variance : float, optional
        User-specified likelihood variance. If `None`, it is set according to the default
        signal-to-noise ratio.
    """
    if likelihood_variance is None:
        noise_variance = variance / SIGNAL_NOISE_RATIO_LIKELIHOOD**2
    else:
        tf.debugging.assert_positive(likelihood_variance)
        noise_variance = tf.cast(likelihood_variance, dtype=gpflow.default_float())

    model.likelihood.variance = gpflow.base.Parameter(
        noise_variance, transform=gpflow.utilities.positive(lower=1e-12)
    )


def _get_kernel(
    variance: TensorType,
    search_space: SearchSpace,
    add_prior_to_lengthscale: bool,
    add_prior_to_variance: bool,
) -> gpflow.kernels.Kernel:
    """
    Build a Matern52 kernel with optional priors.

    Parameters
    ----------
    variance : TensorType
        Kernel variance.
    search_space : SearchSpace
        The domain of input variables.
    add_prior_to_lengthscale : bool
        Whether to add a log-normal prior to the lengthscale.
    add_prior_to_variance : bool
        Whether to add a log-normal prior to the variance.

    Returns
    -------
    gpflow.kernels.Kernel
        The constructed kernel.

    TODO
    ----
    - Expose kernel selection an parameters to the GUI
    - Verify if Mater52 is a good default for classification problems.
    """
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


# End Trieste functions
##


def build_vgp(
    data: Dataset,
    search_space: SearchSpace,
    kernel_priors: bool = True,
    likelihood_variance: float | None = None,
    trainable_likelihood: bool = False,
) -> VGP:
    """
    Build a :class:`~gpflow.models.VGP` following same pattern as build_vgp_classifier,
    but for regression.

    Model is initialized with sensible initial parameters and priors.
    By default it uses :class:`~gpflow.kernels.Matern52` kernel and
    :class:`~gpflow.mean_functions.Constant` mean function in the model.

    Parameters
    ----------
    data : Dataset
        Dataset for estimating empirical variance and mean.
    search_space : SearchSpace
        Search space for scaling kernel parameters.
    kernel_priors : bool, optional
        If True, add priors to kernel parameters (default is True).
    likelihood_variance : float, optional
        Optional fixed noise variance. If None, computed from `SIGNAL_NOISE_RATIO_LIKELIHOOD`.
    trainable_likelihood : bool, optional
        Whether the likelihood variance should be trainable (default False).

    Returns
    -------
    VGP
        A gpflow.models.VGP instance.

    TODO
    ----
        Verify if default initial parameters are good for regression too.
        Expose kernel choice selection.
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


def build_model(init_dataset, search_space, config, problem_type, **kwargs):
    """
    Build a GP model according to the problem type and configuration.

    Parameters
    ----------
    init_dataset : Dataset
        Dataset from the initial design.
    search_space : SearchSpace
        Search space used for model scaling.
    config : dict
        Configuration dictionary with keys:
        - 'variational': bool
        - 'sparse': bool
        - 'trainable_likelihood': bool
        - 'noise_free': bool
        - 'kernel_variance': float
    problem_type : str
        Either 'regression' or 'classification'.
    **kwargs
        Additional parameters.

    Returns
    -------
    Trieste GP model
        One of GaussianProcessRegression, SparseGaussianProcessRegression,
        VariationalGaussianProcess, or SparseVariational depending on config.

    TODO
    ----
        build_svgp by Trieste does not have a noisy version. Need a new function here.

    """

    if problem_type == "classification":
        if config["variational"] and config["sparse"]:
            if config["noise_free"]:
                msg = "build_svgp by Trieste does not have a noisy version"
                warnings.warn(msg)
            gpflow_model = build_svgp(
                init_dataset,
                search_space,
                classification=True,
                # noise_free=config['experiment']['noise_free'],
                trainable_likelihood=config["trainable_likelihood"],
            )
            model = SparseVariational(gpflow_model)
        elif config["variational"] and ~config["sparse"]:
            gpflow_model = build_vgp_classifier(
                init_dataset, search_space, noise_free=config["noise_free"]
            )
            model = VariationalGaussianProcess(gpflow_model)
        else:
            raise Exception(
                "Classification cannot be performed with non-variational GPs"
            )

    elif problem_type == "regression":
        if config["variational"] and config["sparse"]:
            gpflow_model = build_svgp(
                init_dataset,
                search_space,
                classification=False,
                likelihood_variance=config["kernel_variance"],
                trainable_likelihood=config["trainable_likelihood"],
                num_inducing_points=20,
            )
            inducing_point_selector = KMeansInducingPointSelector()
            model = SparseVariational(
                gpflow_model,
                num_rff_features=1000,
                inducing_point_selector=inducing_point_selector,
                optimizer=BatchOptimizer(
                    tf.optimizers.Adam(0.1), max_iter=100, batch_size=50, compile=True
                ),
            )

        elif config["variational"] and ~config["sparse"]:
            gpflow_model = build_vgp(
                init_dataset,
                search_space,
                likelihood_variance=config["kernel_variance"],
                trainable_likelihood=config["trainable_likelihood"],
            )
            model = VariationalGaussianProcess(gpflow_model)

        elif ~config["variational"] and config["sparse"]:
            gpflow_model = build_sgpr(
                init_dataset,
                search_space,
                likelihood_variance=config["kernel_variance"],
                trainable_likelihood=config["trainable_likelihood"],
            )
            inducing_point_selector = KMeansInducingPointSelector()
            model = SparseGaussianProcessRegression(
                gpflow_model,
                num_rff_features=1000,
                inducing_point_selector=inducing_point_selector,
                optimizer=BatchOptimizer(
                    tf.optimizers.Adam(0.1), max_iter=100, batch_size=50, compile=True
                ),
            )

        elif ~config["variational"] and ~config["sparse"]:
            gpflow_model = build_gpr(
                init_dataset,
                search_space,
                likelihood_variance=config["kernel_variance"],
                trainable_likelihood=config["trainable_likelihood"],
            )
            model = GaussianProcessRegression(gpflow_model)

        else:
            raise NotImplementedError("Invalid regression GP configuration.")
    else:
        raise NotImplementedError(
            f"No logic implemented for problem_type {problem_type}"
        )

    return model

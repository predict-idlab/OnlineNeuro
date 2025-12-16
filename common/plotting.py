# common/plotting.py
""" Plotting utilities for visualizing Bayesian Optimization runs with Trieste/OnlineNeuro."""
import os
import warnings
from pathlib import Path
from typing import Any, Callable

import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure, SubFigure
from mpl_toolkits.mplot3d import axes3d
from trieste.acquisition.multi_objective.pareto import Pareto
from trieste.data import Dataset
from trieste.models.interfaces import ProbabilisticModel
from trieste.observer import OBJECTIVE
from trieste.space import SearchSpace

# TODO update functions for more clarity
GRID_RESOLUTION = 0.02
GRID_POINTS_1D = 50  # Number of points for 1D plots
GRID_POINTS_2D = 20  # Number of points for 2D plots (per dimension)

ScalerType = Any
# The following line is necessary for Matplotlib to recognize the 3D projection.
# It is not directly used in the code, so we tell the linter to ignore it.
axes3d.__dir__  # noqa: F401


# TODO HV calculation so that previously calculated values are cached
def custom_cmap() -> LinearSegmentedColormap:
    """
    Create a custom colormap with two colors ('C1', 'C0').

    Returns:
        A Matplotlib LinearSegmentedColormap.
    """
    colors = ["C1", "C0"]
    cmap_name = "custom_cmap"
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=2)
    return cmap


def save_figure(
    fig: Figure | SubFigure,
    save_path: str | Path,
    dpi: int = 300,
    save_svg: bool = False,
) -> None:
    """
    Save the figure to the specified path and closes it.

    Parameters
    ----------
    fig : Figure | SubFigure
        The matplotlib figure or subfigure to save.
    save_path: str | Path
        The base file path to save the figure (excludes the extensiopn).
    dpi: int
        Dots per inch for raster formats (PNG). Defaults to 300.
    save_svg: bool
         If True, saves as SVG; otherwise, saves as PNG. Defaults to False.
    """
    save_path = Path(save_path)

    if save_svg:
        path = save_path.with_suffix(".svg")
        fmt = "svg"
    else:
        path = save_path.with_suffix(".png")
        fmt = "png"

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, format=fmt)
    if isinstance(fig, Figure):
        plt.close(fig)
    elif isinstance(fig, SubFigure):
        plt.close(fig.figure)


def plot_log_reg(
    model: ProbabilisticModel,
    initial_data: Dataset,
    search_space: SearchSpace,
    scaler=None,
    sampled_data=None,
    test_data=None,
    plot_ground_truth=None,
    ground_truth_function=None,
    save_dir: str | Path | None = None,
    count: int = 0,
) -> Axes:
    """
    Plots a 1D logistic regression model with mean predictions and uncertainty.

    Parameters
    ----------
    model : ProbabilisticModel
        The ProbabilisticModel (trieste) model interface to plot predictions from.
    initial_data : trieste.data.Dataset
        The initial dataset (query points and observations).
    search_space : trieste.space.SearchSpace
        The search space definition, used for defining plot limits.
    scaler : ScalerType, optional
        Data scaler used, if any.
    sampled_data : trieste.data.Dataset, optional
        New data points sampled during optimization (acquisition results).
    test_data : trieste.data.Dataset, optional
        Data points used for evaluation (e.g., test set prediction).
    plot_ground_truth : bool, optional
        Whether to plot the true underlying function. Defaults to False.
    ground_truth_function : Callable, optional
        The true function to plot, if available. Must accept (X, noise=0).
    save_dir : str or pathlib.Path, optional
        The directory to save the figure.
    count : int, optional
        The current iteration number, used for the filename. Defaults to 0.

    Raises
    ------
    ValueError
        If `plot_ground_truth` is True but `ground_truth_function` is None.
    """

    if plot_ground_truth and ground_truth_function is None:
        raise ValueError(
            "ground_truth_function must be provided if plot_ground_truth is True."
        )
    ndims = initial_data.query_points.numpy().squeeze().ndim
    if ndims > 1:
        ValueError(f"plot_log_reg expects 1-dimensional inputs, received {ndims}")

    X_init = np.atleast_2d(initial_data.query_points.numpy().squeeze())
    Y_init = np.atleast_2d(initial_data.observations.numpy().squeeze())

    fig, ax = plt.subplots(figsize=(6, 3))

    # Plot initial data
    ax.plot(
        X_init[:, 0],
        Y_init[:, 0],  # By default the first feature
        "ro",
        mew=2,
        label="Initial samples",
    )

    # Plot ground truth if available
    if plot_ground_truth and ground_truth_function:
        x_min = search_space.lower.numpy().squeeze()
        x_max = search_space.upper.numpy().squeeze()
        x_plot = np.linspace(x_min, x_max, num=GRID_POINTS_1D).reshape(-1, 1)

        mean = ground_truth_function(x_plot, noise=0)

        ax.plot(x_plot, mean, color="k", linestyle="--", label="Ground truth")

    if sampled_data:
        X_sampled = sampled_data.query_points
        mean, _ = model.predict(X_sampled)
        ax.plot(
            X_sampled.numpy().squeeze(),
            mean.numpy().squeeze(),
            "rx",
            mew=2,
            label="Sampled acquisition points",
        )

    if test_data:
        X_test = test_data.query_points
        mean, var = model.predict_y(X_test)

        X_test_np = X_test.numpy().squeeze()
        mean_np = mean.numpy().squeeze()
        std_np = np.sqrt(var.numpy().squeeze())

        # Sort for proper filling
        sort_idx = np.argsort(X_test_np)
        X_test_np = X_test_np[sort_idx]
        mean_np = mean_np[sort_idx]
        std_np = std_np[sort_idx]

        ax.plot(X_test_np, mean_np, "C0", lw=2, label="Test prediction mean")
        ax.fill_between(
            X_test_np,
            mean_np - 2 * std_np,
            mean_np + 2 * std_np,
            color="C0",
            alpha=0.2,
            label="Uncertainty (2 std)",
        )

    if hasattr(model.model, "inducing_variable"):
        iv = model.model.inducing_variable
        if hasattr(iv, "Z"):
            Z = iv.Z
            ind_preds, _ = model.predict_y(Z)
            ax.plot(
                Z.numpy().squeeze(),
                ind_preds.numpy().squeeze(),
                "kx",
                mew=2,
                label="Inducing variables",
            )

    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.set_title("Online Logistic Regression")

    if save_dir:
        save_dir = Path(save_dir)
        save_figure(fig, save_dir / f"plot_{count:02d}")

    return ax


def plot_2d_classification(
    model: ProbabilisticModel,
    initial_data: Dataset,
    search_space: SearchSpace,
    scaler: ScalerType = None,
    test_data: Dataset | None = None,
    sampled_data: Dataset | None = None,
    plot_ground_truth: bool = False,
    ground_truth_function: Callable | None = None,
    save_dir: str | Path | None = None,
    count: int = 0,
):
    """
    Plots a 2D input Gaussian Process Classification model
    showing the probability surface P(y=1) in 3D projection.

    Parameters
    ----------
    model : ProbabilisticModel
        The ProbabilisticModel (Trieste) model interface.
    initial_data : trieste.data.Dataset
        The initial dataset.
    search_space : trieste.space.SearchSpace
        The search space definition.
    scaler : ScalerType, optional
        Data scaler used, if any.
    test_data : trieste.data.Dataset, optional
        Test data points and their predictions.
    sampled_data : trieste.data.Dataset, optional
        New data points sampled during optimization.
    plot_ground_truth : bool, optional
        Whether to plot the true underlying classification boundary.
    ground_truth_function : Callable, optional
        The true classification boundary function (must return probabilities).
    save_dir : str or pathlib.Path, optional
        Directory to save figure.
    count : int, optional
        Iteration count.
    """

    if plot_ground_truth:
        assert callable(ground_truth_function)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    x_min = search_space._lower.numpy()
    x_max = search_space._upper.numpy()

    xx, yy = np.meshgrid(
        np.linspace(x_min[0], x_max[0], GRID_POINTS_2D),
        np.linspace(x_min[1], x_max[1], GRID_POINTS_2D),
    )

    X_grid = np.c_[xx.ravel(), yy.ravel()]
    mean, _ = model.predict_y(tf.cast(X_grid, tf.float64))

    try:
        likelihood = model.model.likelihood
        if isinstance(likelihood, gpflow.likelihoods.Bernoulli):
            Z_prob = likelihood.invlink(mean).numpy().squeeze()
        else:
            Z_prob = mean.numpy().squeeze()
    except AttributeError:
        Z_prob = mean.numpy().squeeze()

    Z = Z_prob.reshape(xx.shape)

    ax.plot_surface(xx, yy, Z, cmap=cm.coolwarm, alpha=0.8)

    # Highlighting initial data points
    X_init = initial_data.query_points.numpy()
    Y_init = initial_data.observations.numpy().squeeze()

    # Highlighting initial data points
    locs = Y_init == 1

    ax.scatter(
        X_init[locs, 0],
        X_init[locs, 1],
        Y_init[locs],
        marker="o",
        c="k",
        label="Train-pos",
    )
    ax.scatter(
        X_init[~locs, 0],
        X_init[~locs, 1],
        Y_init[~locs],
        marker="x",
        c="k",
        label="Train-neg",
    )

    # Plotting the sampled data
    if sampled_data and len(sampled_data.query_points) > 0:
        X_sampled = sampled_data.query_points.numpy()

        # Get predictions for height visualization
        mean_sampled, _ = model.predict_y(sampled_data.query_points)
        Y_sampled_pred = mean_sampled.numpy().squeeze()

        ax.scatter(
            X_sampled[:, 0],
            X_sampled[:, 1],
            Y_sampled_pred,
            marker="*",
            c="r",
            label="Sampled acquisition points",
        )
        # Plotting the ground truth boundary
    if plot_ground_truth and ground_truth_function is not None:
        Z_gt = ground_truth_function(X_grid)
        Z_gt = Z_gt.reshape(xx.shape)

        # Contour Z_gt=0.5 projected slightly below the surface
        ax.contour(
            xx,
            yy,
            Z_gt,
            levels=[0.5],
            colors="k",
            linestyles="solid",
            linewidths=2,
            offset=-0.1,
        )

    # Plotting test samples
    if test_data:
        X_test = test_data.query_points
        X_test_np = X_test.numpy()

        mean, _ = model.predict_y(X_test)

        # Predicted probability and labels
        likelihood = model.model.likelihood
        if isinstance(likelihood, gpflow.likelihoods.Bernoulli):
            Z_test_prob = likelihood.invlink(mean).numpy().squeeze()
        else:
            Z_test_prob = mean.numpy().squeeze()

        preds = tf.math.round(Z_test_prob).numpy().astype(bool).squeeze()

        # We plot the prediction probability Z value (height)
        ax.scatter(
            X_test_np[preds, 0],
            X_test_np[preds, 1],
            Z_test_prob[preds],
            c="C0",
            marker="o",
            label="Pos-test Prediction",
        )
        ax.scatter(
            X_test_np[~preds, 0],
            X_test_np[~preds, 1],
            Z_test_prob[~preds],
            c="C1",
            marker="x",
            label="Neg-test Prediction",
        )

    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("P(Y=1)")
    ax.set_title("2D Classification Surface Prediction")
    ax.legend()

    if save_dir:
        save_dir = Path(save_dir)
        save_figure(fig, save_dir / f"plot_{count:02d}")


def plot_predictions_2d(
    model: ProbabilisticModel,
    initial_data: Dataset,
    search_space: SearchSpace,
    scaler: ScalerType | None = None,
    test_data: Dataset | None = None,
    sampled_data: Dataset | None = None,
    plot_ground_truth: bool = False,
    ground_truth_function: Callable | None = None,
    save_dir: str | Path | None = None,
    count: int = 0,
):
    """
    Plots predictions for a 2D input, 1D output regression task (e.g., Rosenbrock),
    visualizing the mean prediction surface.

    Parameters
    ----------
    model : ProbabilisticModel
        The Trieste model interface.
    initial_data : trieste.data.Dataset
        Initial dataset.
    search_space : trieste.space.SearchSpace
        Search space definition.
    scaler : ScalerType, optional
        Data scaler (currently unused in core plotting logic).
    test_data : trieste.data.Dataset, optional
        Test points and their predictions.
    sampled_data : trieste.data.Dataset, optional
        Sampled points from the acquisition function.
    plot_ground_truth : bool, optional
        Whether to plot ground truth (not fully implemented).
    ground_truth_function : Callable, optional
        The true function.
    save_dir : str or pathlib.Path, optional
        Directory to save figure.
    count : int, optional
        Iteration count.

    Notes
    ----------
        Linting has issues detecting the axes3D type.
        Ignore the warnings.
    """
    if plot_ground_truth:
        assert callable(ground_truth_function)

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111, projection="3d")
    ax = [ax1]

    x_min, y_min = search_space.lower.numpy()
    x_max, y_max = search_space.upper.numpy()

    # Generate grid
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, GRID_POINTS_2D),
        np.linspace(y_min, y_max, GRID_POINTS_2D),
    )
    X_grid = np.c_[xx.ravel(), yy.ravel()]

    # Prediction
    Z_tf, _ = model.predict_y(tf.cast(X_grid, tf.float64))

    Z = Z_tf.numpy().squeeze()
    Z = Z.reshape(xx.shape)

    ax[0].plot_surface(xx, yy, Z, alpha=0.8, cmap=cm.viridis)

    # Plot Initial Data
    if initial_data is not None:
        X_init = initial_data.query_points.numpy()
        Y_init = initial_data.observations.numpy().squeeze()
        ax[0].scatter(
            X_init[:, 0],
            X_init[:, 1],
            Y_init,
            marker="o",
            c="k",
            label="Initial samples",
            s=30,
        )

    # Plot Sampled Data
    if sampled_data:
        X_samps = sampled_data.query_points
        Y_samps, _ = model.predict_y(X_samps)
        ax[0].scatter(
            X_samps[:, 0].numpy(),
            X_samps[:, 1].numpy(),
            Y_samps[:, 0].numpy(),
            marker="x",
            c="r",
            label="Sampled Acquisition Points",
            s=30,
        )

    # Plot Test Data
    if test_data:
        X_test = test_data.query_points
        Y_test_mean, _ = model.predict_y(X_test)

        ax[0].scatter(
            X_test[:, 0].numpy(),
            X_test[:, 1].numpy(),
            Y_test_mean[:, 0].numpy(),
            c="C0",
            marker="s",
            s=30,
            label="Test Predictions",
        )

    ax[0].set_xlabel("X1")
    ax[0].set_ylabel("X2")
    ax[0].set_zlabel("Predicted Y")
    ax[0].set_title("2D Prediction Surface")
    ax[0].legend()

    if save_dir:
        save_dir = Path(save_dir)
        save_figure(fig, save_dir / f"plot_{count:02d}")


def calculate_reference_point(observations: tf.Tensor) -> tf.Tensor:
    """
    Calculates the reference point for hypervolume calculation based on the Pareto front.
    (Assumes minimization objectives where the origin is the ideal point).

    The reference point is calculated as the maximum point on the front plus
    a small buffer based on the front's range.

    Parameters
    ----------
    observations : tf.Tensor
        Tensor of objective observations (N, D).

    Returns
    -------
    tf.Tensor
        The calculated reference point (D,).
    """
    pareto_obj = Pareto(observations)
    front = pareto_obj.front

    f = tf.math.reduce_max(front, axis=0) - tf.math.reduce_min(front, axis=0)

    N_front = tf.cast(tf.shape(front)[0], f.dtype)
    ref = tf.math.reduce_max(front, axis=0) + 2 * f / N_front

    return ref


def log_hv(obs: tf.Tensor, ref_point: tf.Tensor) -> float:
    """
    Calculates the log base 10 Hypervolume indicator relative to a reference point.

    Parameters
    ----------
    obs : tf.Tensor
        Tensor of observations (N, D).
    ref_point : tf.Tensor
        Reference point tensor (D,).

    Returns
    -------
    float
        log10(Hypervolume).
    """
    obs_hv = Pareto(obs).hypervolume_indicator(ref_point)
    return np.log10(obs_hv)


def plot_pareto_2d(
    model: ProbabilisticModel,
    initial_data: Dataset,
    search_space: SearchSpace,
    scaler: ScalerType | None = None,
    sampled_data: Dataset | None = None,
    plot_ground_truth: bool = False,
    ground_truth_function: Callable | None = None,
    save_dir: str | Path | None = None,
    count: int = 0,
):
    """
    Plots results for 2D input, 2D output multiobjective optimization.

    This generates a 4-panel figure showing:
    1. Objective 1 prediction contour (input space).
    2. Objective 2 prediction contour (input space).
    3. Pareto front (objective space).
    4. Hypervolume history over iterations.

    Parameters
    ----------
    model : ProbabilisticModel
        The Trieste model interface (expected to be a model stack).
    initial_data : trieste.data.Dataset
        Initial dataset.
    search_space : trieste.space.SearchSpace
        Search space definition.
    scaler : ScalerType, optional
        Data scaler (currently unused in core plotting logic).
    sampled_data : trieste.data.Dataset, optional
        Sampled points from the acquisition function, including observations.
    plot_ground_truth : bool, optional
        Whether to plot ground truth (currently unsupported).
    ground_truth_function : Callable, optional
        The true function (currently unsupported).
    save_dir : str or pathlib.Path, optional
        Directory to save figure.
    count : int, optional
        Iteration count.
    """

    fig, ax = plt.subplots(figsize=(12, 10), ncols=2, nrows=2)
    ax = ax.ravel()

    X_init = initial_data.query_points.numpy()
    Y_init = initial_data.observations.numpy()

    X_samp = sampled_data.query_points.numpy() if sampled_data else np.array([])

    # --- Panel 0 & 1: Prediction Contours (in input space) ---
    x_min, y_min = search_space.lower.numpy()
    x_max, y_max = search_space.upper.numpy()

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, GRID_POINTS_2D),
        np.linspace(y_min, y_max, GRID_POINTS_2D),
    )
    X_grid = tf.cast(np.c_[xx.ravel(), yy.ravel()], tf.float64)

    assert isinstance(X_grid, tf.Tensor), "X_grid must be a tf.Tensor"

    Z_mean, _ = model.predict(X_grid)
    Z1 = Z_mean.numpy()[:, 0].reshape(*xx.shape)
    Z2 = Z_mean.numpy()[:, 1].reshape(*xx.shape)

    ax[0].contourf(xx, yy, Z1, 40, cmap="viridis", zorder=1, alpha=0.8)
    ax[1].contourf(xx, yy, Z2, 40, cmap="viridis", zorder=1, alpha=0.8)

    # Plot initial and sampled points on contours
    for a in ax[:2]:
        a.scatter(
            X_init[:, 0],
            X_init[:, 1],
            zorder=2,
            marker="o",
            c="orange",
            label="Initial samples",
        )
        if sampled_data:
            a.scatter(
                X_samp[:, 0],
                X_samp[:, 1],
                zorder=3,
                marker="D",
                c="red",
                label="Sampled point",
            )
        a.set_xlabel("X1")
        a.set_ylabel("X2")

    ax[0].set_title("Objective 1 Prediction Mean")
    ax[1].set_title("Objective 2 Prediction Mean")

    # --- Combining observed data for Pareto/HV calculation ---
    Y_observed = Y_init
    if sampled_data and sampled_data.observations.shape[0] > 0:
        Y_observed = np.vstack([Y_init, sampled_data.observations.numpy()])

    Y_observed_tf = tf.Tensor(Y_observed, dtype=tf.float64)

    # --- Panel 2: Pareto Front (in objective space) ---
    front = Pareto(Y_observed_tf).front.numpy()

    N_init = len(Y_init)

    ax[2].scatter(
        Y_init[:, 0],
        Y_init[:, 1],
        c="C0",
        marker="o",
        label=f"Initial Samples (N={N_init})",
    )

    if sampled_data and sampled_data.observations.shape[0] > 0:
        Y_samp_obs = sampled_data.observations.numpy()
        ax[2].scatter(
            Y_samp_obs[:, 0],
            Y_samp_obs[:, 1],
            c="purple",
            marker="D",
            label="Observed Sampled Points",
        )

    # Plot Pareto Front line
    if front.size > 0:
        order = np.argsort(front[:, 0])
        sorted_front = front[order]
        ax[2].plot(
            sorted_front[:, 0],
            sorted_front[:, 1],
            c="r",
            linestyle="--",
            label="Estimated Pareto Front",
        )

    # Set common VLMOP2 limits
    ax[2].set_xlim(0, 1.2)
    ax[2].set_ylim(0, 1.2)
    ax[2].set_xlabel("Objective #1")
    ax[2].set_ylabel("Objective #2")
    ax[2].set_title("Pareto Front in Objective Space")

    # --- Panel 3: Hypervolume History ---

    ref_point = calculate_reference_point(Y_observed_tf)

    _idxs = np.arange(1, len(Y_observed) + 1)

    # Calculate cumulative HV
    log_vol = [log_hv(Y_observed_tf[:i, :], ref_point) for i in _idxs]

    ax[3].plot(_idxs, log_vol, color="C1", label="Log hypervolume (Cumulative)")

    ax[3].axvline(x=N_init, color="C2", linestyle=":", label="End of Initial Samples")

    ax[3].grid(True, linestyle="--")
    ax[3].set_xlabel("Cumulative Data Points (N)")
    ax[3].set_ylabel("Log (Hypervolume)")
    ax[3].set_title("Hypervolume Progress")

    # Final layout adjustments
    ax[0].legend(loc="upper right")
    ax[1].legend(loc="upper right")
    ax[2].legend(loc="lower left")
    ax[3].legend(loc="lower right")

    fig.suptitle(f"Multiobjective Optimization Step #{count}", fontsize=14)
    # fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    if save_dir:
        save_dir = Path(save_dir)
        save_figure(fig, save_dir / f"plot_{count:02d}")


def plot_classification_2d_slices(
    model,
    initial_data: Dataset,
    search_space: SearchSpace,
    scaler: ScalerType = None,
    sampled_data: Dataset | None = None,
    plot_ground_truth: bool = False,
    ground_truth_function: Callable | None = None,
    save_dir: str | Path | None = None,
    count: int = 0,
):
    """
    Plots predictions for a high-dimensional binary classification problem showing 2D slices.

    For each unique pair of input features (X_i, X_j), a 2D contour plot is generated
    by fixing all other features at the midpoint of the search space.

    Parameters
    ----------
    model : ProbabilisticModel
        The Trieste model interface.
    initial_data : trieste.data.Dataset
        The initial dataset.
    search_space : trieste.space.SearchSpace
        The search space definition.
    scaler : ScalerType, optional
        Data scaler (if used).
    sampled_data : trieste.data.Dataset, optional
        New data points sampled during optimization.
    plot_ground_truth : bool, optional
        Whether to plot ground truth (currently unsupported).
    ground_truth_function : Callable, optional
        The true function (currently unsupported).
    save_dir : str or pathlib.Path, optional
        Directory to save figure.
    count : int, optional
        Iteration count.

    Warnings
    --------
    If `num_vars` < 2, the plot is skipped. Ground truth plotting is complex
    and currently skipped.
    """
    if plot_ground_truth:
        assert callable(ground_truth_function)

    mins_ = search_space.lower.numpy()
    maxs_ = search_space.upper.numpy()
    num_vars = len(mins_)

    if num_vars < 2:
        warnings.warn(
            "Cannot plot 2D slices for problems with fewer than 2 dimensions."
        )
        return

    # Generate unique pairs of indices
    counts = np.arange(num_vars)
    feat_pairs = [(i, j) for i in counts for j in counts if i < j]
    num_feat_pairs = len(feat_pairs)

    # Determine layout (max 4 columns)
    N_COLS = min(num_feat_pairs, 4)
    N_ROWS = int(np.ceil(num_feat_pairs / N_COLS))

    fig, ax = plt.subplots(figsize=(N_COLS * 5, N_ROWS * 5), ncols=N_COLS, nrows=N_ROWS)
    ax = np.atleast_1d(ax).ravel()

    mean_values = (mins_ + maxs_) / 2
    norm = Normalize(vmin=0, vmax=1)

    # Data points (in search space coordinates)
    X_init = initial_data.query_points.numpy()
    Y_init = initial_data.observations.numpy().squeeze()
    X_samp = sampled_data.query_points.numpy() if sampled_data else np.array([])

    cmap_binary = custom_cmap()

    for i, (x_ix, y_ix) in enumerate(feat_pairs):
        x_min, y_min = mins_[x_ix], mins_[y_ix]
        x_max, y_max = maxs_[x_ix], maxs_[y_ix]

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, GRID_POINTS_2D),
            np.linspace(y_min, y_max, GRID_POINTS_2D),
        )

        # 1. Construct high-D input tensor (other features fixed at mean)
        X_grid_flat = np.c_[xx.ravel(), yy.ravel()]
        N_grid = X_grid_flat.shape[0]

        inputs = np.tile(mean_values, (N_grid, 1))
        inputs[:, x_ix] = X_grid_flat[:, 0]
        inputs[:, y_ix] = X_grid_flat[:, 1]

        inputs_tf = tf.cast(inputs, tf.float64)

        # 2. Prediction
        Z_mean, _ = model.predict_y(inputs_tf)
        Z_prob = gpflow.likelihoods.Bernoulli().invlink(Z_mean).numpy().squeeze()
        Z = Z_prob.reshape(xx.shape)

        ax[i].contourf(xx, yy, Z, alpha=0.8, cmap=cm.coolwarm, norm=norm)
        ax[i].set_ylabel(f"Feature {y_ix}")
        ax[i].set_xlabel(f"Feature {x_ix}")
        ax[i].set_title(f"Slice: X{x_ix} vs X{y_ix}")

        # 4. Plot Initial Data points
        ax[i].scatter(
            X_init[:, x_ix],
            X_init[:, y_ix],
            c=Y_init,
            marker="o",
            edgecolors="k",
            s=60,
            cmap=cmap_binary,
            zorder=2,
        )

        # 5. Plot Sampled Data points
        if X_samp.size > 0:
            ax[i].scatter(
                X_samp[:, x_ix],
                X_samp[:, y_ix],
                c="r",
                marker="X",
                s=80,
                label="Sampled acquisition point",
                zorder=3,
            )

    # Clean up empty subplots
    for j in range(num_feat_pairs, N_ROWS * N_COLS):
        if j < len(ax):
            fig.delaxes(ax[j])

    # Add legend to the first subplot only
    if len(ax) > 0:
        ax[0].legend(loc="lower right")

    fig.suptitle(f"Nerve Block 2D Slices Step #{count}", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    if save_dir:
        save_dir = Path(save_dir)
        save_figure(fig, save_dir / f"plot_{count:02d}")


def update_plot(
    bo,
    initial_data: Dataset,
    search_space: SearchSpace,
    scaler,
    sampled_data: Dataset = None,
    test_data: Dataset = None,
    plot_ground_truth: bool = False,
    ground_truth_function=None,
    count: int = 0,
    *args,
    **kwargs,
) -> None:
    """Update/Save the plot based on the current state of the Bayesian Optimizer.
    Args:
        bo: BayesianOptimizer object.
        initial_data: The initial dataset.
        search_space: The search space definition.
        scaler: The data scaler, if used.
        sampled_data: New data points sampled during optimization.
        test_data: Data points to evaluate (normally test).
        plot_ground_truth: Whether to plot the true underlying function.
        ground_truth_function: The true function to plot, if available.
        count: The current iteration number.
        **kwargs: Additional keyword arguments for specific plotters.

    """

    # TODO implement and improve current_plotting functions
    # TODO Include inverse scaling for features when bo.scaler is not None
    #      Partially done for some methods
    observer_name = str(bo._observer).lower()
    save_dir = Path("figures") / observer_name
    os.makedirs(save_dir, exist_ok=True)

    common_args = {
        "model": bo._models[OBJECTIVE],
        "initial_data": initial_data,
        "search_space": search_space,
        "scaler": scaler,
        "sampled_data": sampled_data,
        "plot_ground_truth": plot_ground_truth,
        "ground_truth_function": ground_truth_function,
        "save_dir": save_dir,
        "count": count,
        **kwargs,
    }
    plot_functions = {
        "log_reg": plot_log_reg,
        "circle_classification": plot_2d_classification,
        "axonsim_nerve_block": plot_classification_2d_slices,
        "rosenbruck": plot_predictions_2d,
        "axon_single": plot_predictions_2d,
        "axon_double": plot_predictions_2d,
        "axon_threshold": plot_predictions_2d,
        "vlmop2": plot_pareto_2d,
        "multiobjective": plot_pareto_2d,
    }

    plot_func = plot_functions.get(observer_name)
    if plot_func:
        plot_func(**common_args)
    else:
        warnings.warn(f"Plot function not implemented for '{observer_name}' problems.")

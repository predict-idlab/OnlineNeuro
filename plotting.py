from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
import warnings
import tensorflow as tf
from trieste.observer import OBJECTIVE
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import gpflow
from matplotlib import cm
import os
import itertools
from trieste.experimental.plotting import plot_bo_points, plot_function_2d
import matplotlib.pyplot as plt
import seaborn as sns
from trieste.models.interfaces import TrainableModelStack
from trieste.acquisition.multi_objective.pareto import Pareto
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

GRID_RESOLUTION = 0.02
#TODO, change grid resolution to DATAPOINTS
#So that sampling spaces are the same for each feature (features are not normalized)

plt.style.use('ggplot')

# TODO HV calculation so that previously calculated values are cached
def custom_cmap():
    colors = ['C1', 'C0']
    cmap_name = 'custom_cmap'
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=2)
    return cmap


def plot_model(model, initial_data, search_space=None, scaler=None, test_data=None, sampled_data=None,
               plot_ground_truth=None, ground_truth_function=None, save_dir=None, count=0):
    """
    Plot a two feature 1 output model
    @param scaler:
    @param model: Trieste model
    @param initial_data: Trieste Dataset
    @param search_space: Trieste search space
    @param test_data: Data points to evaluate (normally test)
    @param sampled_data: Data points to evaluate (normally the incremental train)
    @param plot_ground_truth: boolean to decide whether ground_truth is plotted
    @param ground_truth_function: the generator function of the problem (available for some toy problems)
    @param save_dir: path to save figure
    @param count: numeric value to include in the figure name
    @return:
    """
    if plot_ground_truth:
        assert ground_truth_function is not None
        assert search_space is not None

    plt.figure(figsize=(6, 3))
    plt.plot(initial_data[0][:, 0],
             initial_data[1][:, 0],
             'ro', mew=2,
             label='Initial samples')

    if plot_ground_truth:
        x_min = search_space._lower.numpy()
        x_max = search_space._upper.numpy()
        x = np.linspace(x_min, x_max, num=50)
        mean = ground_truth_function(x, noise=0)
        plt.plot(x,
                 mean,
                 color='k', linestyle='--',
                 label='Ground truth')

    if sampled_data:
        mean, var = model.predict(sampled_data[0])
        plt.plot(sampled_data[0][:, 0],
                 mean,
                 'rx', mew=2, label='Sampled datapoints')

    if test_data:
        mean, var = model.predict_y(test_data[0])
        plt.plot(test_data[0][:, 0], mean, 'C0', lw=2, label='Prediction')
        plt.fill_between(test_data[0][:, 0],
                         mean[:, 0] - 2 * np.sqrt(var[:, 0]),
                         mean[:, 0] + 2 * np.sqrt(var[:, 0]),
                         color='C0', alpha=0.2, label='Uncertainty')

    iv = getattr(model, 'inducing_variable', None)
    if iv is not None:
        ind_preds, _ = model.predict_y(iv.Z)
        plt.plot(iv.Z[:, 0], ind_preds, 'rx', mew=2, label='Inducing variables')

    plt.ylim(-0.5, 1.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Online \n Logistic regression')
    if save_dir is None:
        plt.savefig(f'./figures/plot_{count:02}.png')
    else:
        plt.savefig(f'{save_dir}/plot_{count:02}.png')
    plt.close()


def plot_circle(model, initial_data, search_space=None, scaler=None, test_data=None, sampled_data=None, plot_ground_truth=None,
                ground_truth_function=None, save_dir=None, count=0):
    if plot_ground_truth:
        assert callable(ground_truth_function)
        assert search_space is not None
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    if search_space:
        x_min, y_min = search_space._lower.numpy()
        x_max, y_max = search_space._upper.numpy()
        xx, yy = np.meshgrid(np.arange(x_min, x_max+GRID_RESOLUTION, GRID_RESOLUTION),
                             np.arange(y_min, y_max+GRID_RESOLUTION, GRID_RESOLUTION))

        Z, Zvar = model.predict_y(np.c_[xx.ravel(), yy.ravel()])
        Z = gpflow.likelihoods.Bernoulli().invlink(Z).numpy().squeeze()
        Z = Z.reshape(xx.shape)
        ax.plot_surface(xx, yy, Z, cmap=cm.coolwarm)

    x = initial_data[0].numpy()
    locs = (initial_data[1] == 1).numpy().squeeze()

    ax.scatter(x[locs, 0], x[locs, 1], marker='o', c='k', label='Train-pos')
    ax.scatter(x[~locs, 0], x[~locs, 1], marker='x', c='k', label='Train-neg')

    if plot_ground_truth:
        x_min, y_min = search_space._lower.numpy()
        x_max, y_max = search_space._upper.numpy()
        xx, yy = np.meshgrid(np.arange(x_min, x_max+GRID_RESOLUTION, GRID_RESOLUTION),
                             np.arange(y_min, y_max+GRID_RESOLUTION, GRID_RESOLUTION))

        Z = model.predict_y(np.c_[xx.ravel(), yy.ravel()])
        Z = tf.math.round(Z[0]).numpy().squeeze()
        Z = Z.reshape(xx.shape)
        ax.contour(xx, yy, Z, colors='k', linestyles='solid', linewidths=1)

    if sampled_data:
        ax.scatter(sampled_data[0][:, 0], sampled_data[0][:, 1],
                   marker='x', c='r', label='Sampled datapoints')

    if test_data:
        mean, var = model.predict_y(test_data[0])
        mean = gpflow.likelihoods.Bernoulli().invlink(mean).numpy().squeeze()
        preds = tf.math.round(mean).numpy()
        corr_bool = (preds > 0).squeeze()

        ax.scatter(test_data[0].numpy()[corr_bool, 0], test_data[0].numpy()[corr_bool, 1], c='C0', marker='o',
                   label='Pos-test')
        ax.scatter(test_data[0].numpy()[~corr_bool, 0], test_data[0].numpy()[~corr_bool, 1], c='C1', marker='x',
                   label='Neg-test')

    ax.legend()
    # ax.set_xlim(-1, 1)
    # ax.set_ylim(-1, 1)
    if save_dir is None:
        plt.savefig(f'./figures/plot_{count:02}.png')
    else:
        plt.savefig(f'{save_dir}/plot_{count:02}.png')
    plt.close()


def plot_surface(model, initial_data, search_space=None, scaler=None,
                 test_data=None, sampled_data=None, plot_ground_truth=None,
                 ground_truth_function=None, feasible_region=None, save_dir=None, count=0):
    """

    @param scaler:
    @param model: Model from Trieste. TODO in some cases this may be a list with single output.
    @param initial_data:
    @param search_space:
    @param scaler:
    @param test_data:
    @param sampled_data:
    @param plot_ground_truth:
    @param ground_truth_function:
    @param feasible_region:
    @param save_dir:
    @param count:
    @return:
    """
    if plot_ground_truth:
        assert callable(ground_truth_function)
        assert search_space is not None

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax = [ax1]

    if search_space:
        x_min, y_min = search_space._lower.numpy()
        x_max, y_max = search_space._upper.numpy()
        xx, yy = np.meshgrid(np.arange(x_min, x_max+GRID_RESOLUTION, GRID_RESOLUTION),
                             np.arange(y_min, y_max+GRID_RESOLUTION, GRID_RESOLUTION))
        Z, Zvar = model.predict_y(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.numpy().squeeze()
        Z = Z.reshape(xx.shape)
        Zvar = Zvar.numpy().squeeze()
        Zvar = Zvar.reshape(xx.shape)
        ax[0].plot_surface(xx, yy, Z, alpha=0.8)
        # ax[1].contourf(xx, yy, Zvar, alpha=0.8)

    if initial_data is not None:
        ax[0].scatter(initial_data[0][:, 0], initial_data[0][:, 1],
                      initial_data[1][:, 0], marker='o', c='k', label='Train-pos')

    if feasible_region:
        # TODO
        pass

    if sampled_data:
        y_samps, _ = model.predict_y(sampled_data[0])
        ax[0].scatter(sampled_data[0][:, 0], sampled_data[0][:, 1],
                      y_samps, marker='x', c='r', label='Sampled-Points')

    if test_data:
        mean, var = model.predict_y(test_data[0])
        #TODO, check if this is correct
        preds = tf.math.round(mean).numpy()

        ax[0].scatter(test_data[0][:, 0], test_data[0][:, 1], preds, c='k', marker='o', label='Samples added')

    ax[0].legend()
    if save_dir is None:
        plt.savefig(f'./figures/plot_{count:02}.png')
    else:
        plt.savefig(f'{save_dir}/plot_{count:02}.png')
    plt.close()

def calculate_reference_point(observations):
    pareto_obj = Pareto(observations)
    front = pareto_obj.front
    f = tf.math.reduce_max(front, axis=-2) - tf.math.reduce_min(front, axis=-2)
    ref = tf.math.reduce_max(front, axis=-2) + 2 * f / tf.cast(tf.shape(front)[-2], f.dtype)
    return ref

def log_hv(obs, ref_point):
    obs_hv = Pareto(obs).hypervolume_indicator(ref_point)
    return np.log10(obs_hv)

def plot_pareto_2d(model, initial_data, search_space=None,
                   scaler=None, test_data=None, sampled_data=None, plot_ground_truth=None,
                   ground_truth_function=None, save_dir=None, count=0):

    if plot_ground_truth:
        assert callable(ground_truth_function)
        assert search_space is not None

    fig, ax = plt.subplots(figsize=(10, 10), ncols=2, nrows=2)
    ax = ax.ravel()

    if search_space:
        x_min, y_min = search_space._lower.numpy()
        x_max, y_max = search_space._upper.numpy()
        xx, yy = np.meshgrid(np.arange(x_min, x_max+GRID_RESOLUTION, GRID_RESOLUTION),
                             np.arange(y_min, y_max+GRID_RESOLUTION, GRID_RESOLUTION))
        #if isinstance(model, TrainableModelStack):
        Z, Zvar = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.numpy()
        Zvar = Zvar.numpy()
        #else:
        #    raise Exception("Case not defined (If Not TrainableModelStack")
        ax[0].contour(xx, yy, Z[:, 0].reshape(*xx.shape), 40, zorder=1, alpha=1)
        ax[1].contour(xx, yy, Z[:, 1].reshape(*xx.shape), 40, zorder=1, alpha=1)

    ax[0].scatter(initial_data[0][:, 0], initial_data[0][:, 1], zorder=2, marker='o', c='orange', label='Initial samples')
    ax[1].scatter(initial_data[0][:, 0], initial_data[0][:, 1], zorder=2, marker='o', c='orange', label='Initial samples')

    if sampled_data:
        ax[0].scatter(sampled_data[0][:, 0], sampled_data[0][:, 1],
                      zorder=1, marker='o', c='red', label='Sampled-data')
        ax[1].scatter(sampled_data[0][:, 0], sampled_data[0][:, 1],
                      zorder=1, marker='o', c='red', label='Sampled-data')

    if test_data:
        #if isinstance(TrainableModelStack):
        Z, Zvar = model.predict(test_data[0])
        Z = Z.numpy()
        Zvar = Zvar.numpy()
        #else:
        #    raise Exception("Case not defined (If Not TrainableModelStack")
        # TODO, for single models with two outputs the predict may be different, verify this
        ax[0].scatter(test_data[0][:, 0], test_data[0][:, 1], Z[:, 0], zorder=1, c='k', marker='x', label='Test samples')
        ax[1].scatter(test_data[0][:, 0], test_data[0][:, 1], Z[:, 1], zorder=1, c='k', marker='x', label='Test samples')

    #TODO, probably a caching would be handy to not have to recompute all the hlv values
    #TODO. Note: a bit arbitrary how trieste select the ref point, verify if there are other methods. In general
    # it should be fine as the ref. value doesn't change when comparing vs other models/techniques, or acrros incremental
    # calculations of the HV

    Z_init, Zvar_init = model.predict(initial_data[0])
    Z_samp, Zvar_init = model.predict(sampled_data[0])

    Z_init = Z_init.numpy()
    Z_samp = Z_samp.numpy()

    Z_stack = np.vstack([Z_init, Z_samp])

    front = Pareto(Z_stack).front
    ax[2].scatter(Z_init[:, 0], Z_init[:, 1], c='C0', marker='o', label='Initial \n sample')
    ax[2].scatter(Z_samp[:, 0], Z_samp[:, 1], c='purple', marker='o', label='Sampled-data')
    # Just for the toy problem, other problems have ranges beyond these limits.
    ax[2].set_xlim(0, 1.2)
    ax[2].set_ylim(0, 1.2)

    order = tf.argsort(front[:, 0])
    sorted_front = tf.gather(front, order).numpy()
    ax[2].plot(sorted_front[:, 0], sorted_front[:, 1], c='r', label="Pareto front")

    ax[2].set_xlabel("Objective #1")
    ax[2].set_xlabel("Objective #2")

    ref_point = calculate_reference_point(initial_data[1].numpy())

    _idxs = np.arange(1, len(Z_stack) + 1)
    log_vol = [log_hv(Z_stack[:i, :], ref_point) for i in _idxs]
    ax[3].plot(_idxs, log_vol, color='C1', label='Neg. Log hypervolume')
    ax[3].axvline(x=len(initial_data[0]), color='C2', label='Initial samples')
    ax[3].grid()
    ax[3].set_xlabel("Data points")
    ax[3].set_ylabel("Log (HV)")
    ax[3].grid()

    ax[0].legend()
    ax[1].legend()
    ax[2].legend(loc='lower left')
    ax[3].legend()

    ax[0].set_title("Objective 1")
    ax[1].set_title("Objective 2")
    ax[2].set_title("Pareto front")
    ax[3].set_title("Hypervolume")
    fig.suptitle(f"Multiobjective optimization Step #{count}")
    fig.tight_layout()

    if save_dir is None:
        plt.savefig(f'./figures/plot_{count:02}.png')
    else:
        plt.savefig(f'{save_dir}/plot_{count:02}.png')
    plt.close()

def plot_nerve_block(model, initial_data, search_space, scaler=None,
                     test_data=None, sampled_data=None, plot_ground_truth=None,
                     ground_truth_function=None, save_dir=None, count=0):
    """
    Nerve_block is a binary output but has more inputs.
    Here I implement a simple 2 inputs 1 output plot (similar to circle), in which the first 2 variables
    are plotted and for the rest the central value of the search_space is taken for the rest of the features.

    As a side plot, a PCA is included.
    @param scaler:
    """

    if plot_ground_truth:
        assert callable(ground_truth_function)
        assert search_space is not None

    num_vars = len(search_space._lower.numpy())
    counts = np.arange(num_vars)
    feat_pairs = {frozenset((a, b)) for a in counts for b in counts if a != b}
    feat_pairs = [tuple(pair) for pair in feat_pairs]
    num_feat_pairs = len(feat_pairs)

    fig, ax = plt.subplots(figsize=(num_feat_pairs*3, 6), ncols=num_feat_pairs, nrows=2)
    ax = ax.ravel()

    mins_ = search_space._lower.numpy()
    maxs_ = search_space._upper.numpy()

    mean_values = (mins_ + maxs_) / 2
    norm = Normalize(vmin=0, vmax=1)

    for i in range(num_feat_pairs):
        x_ix = feat_pairs[i][0]
        y_ix = feat_pairs[i][1]

        x_min, y_min = mins_[x_ix], mins_[y_ix]
        x_max, y_max = maxs_[x_ix], maxs_[y_ix]

        xx, yy = np.meshgrid(np.arange(x_min, x_max+GRID_RESOLUTION, GRID_RESOLUTION),
                             np.arange(y_min, y_max+GRID_RESOLUTION, GRID_RESOLUTION))

        x_y_cols = np.c_[xx.ravel(), yy.ravel()]
        mean_cols = np.ones((x_y_cols.shape[0], mean_values.shape[0])) * mean_values
        inputs = np.zeros((x_y_cols.shape[0], mean_values.shape[0]))
        other_ixs = [i for i in range(mean_values.shape[0]) if i not in [x_ix, y_ix]]
        inputs[:, x_ix] = x_y_cols[:, 0]
        inputs[:, y_ix] = x_y_cols[:, 1]
        for ix in other_ixs:
            inputs[:, ix] = mean_cols[:, ix]

        Z, Zvar = model.predict_y(inputs)
        Z = gpflow.likelihoods.Bernoulli().invlink(Z)
        Z = Z.numpy()
        Z = Z.squeeze().reshape(xx.shape)

        if scaler:
            xx = scaler.inverse_transform_mat(xx, ix=x_ix)
            yy = scaler.inverse_transform_mat(yy, ix=y_ix)

        ax[i].contourf(xx, yy, Z, alpha=0.8, cmap=cm.coolwarm,
                       norm=norm
                       )
        ax[i].set_ylabel(f"feat_{y_ix}")
        ax[i].set_xlabel(f"feat_{x_ix}")
        ax[i].set_title(f"{i}")

    x = initial_data[0].numpy()
    y = initial_data[1].numpy()

    if scaler:
        x = scaler.inverse_transform(x)

    for i in range(num_feat_pairs):
        x_ix = feat_pairs[i][0]
        y_ix = feat_pairs[i][1]
        ax[i].scatter(x[:, x_ix], x[:, y_ix], c=y, marker='*', edgecolors='k', s=60)

    if plot_ground_truth:
        #TBD
        raise Exception("Ground truth not implemented for Nerve block (yet). This may work as a precomputed result")

    if sampled_data:
        samp_data = sampled_data[0]
        if scaler:
            samp_data = scaler.inverse_transform(samp_data)
        for i in range(num_feat_pairs):
            x_ix = feat_pairs[i][0]
            y_ix = feat_pairs[i][1]
            ax[i].scatter(samp_data[:, x_ix],
                          samp_data[:, y_ix],
                          c='k',
                          marker='*',
                          s=50, label='Sampled data'
                          )
        # ax[1].scatter(sampled_data[0][:, x_ix], sampled_data[0][:, y_ix],
        #               marker='x', c='r', label='Sampled datapoints')
    #ax[0].legend()
    fig.tight_layout()

    if save_dir is None:
        plt.savefig(f'./figures/plot_{count:02}.png')
    else:
        plt.savefig(f'{save_dir}/plot_{count:02}.png')
    plt.close()

def update_plot(bo, initial_data=None, sampled_data=None, test_data=None, plot_ground_truth=None, ground_truth_function=None,
                count=0, *args, **vargs):
    """
    @param bo: BayesianOptimizer object
    @param initial_data: Initial data for plotting
    @param sampled_data: Sampled data for plotting
    @param test_data: Test data for plotting
    @param plot_ground_truth: Boolean to indicate whether the true labels should be plotted. It requires passing a valid
            search space and ground_truth_function.
    @param ground_truth_function: Function that generates the ground truth labels. Only available for toy problems.
    @param count: Counter for saving plot with unique names
    @param args: Additional positional arguments
    @param vargs: Additional keyword arguments
    @return: None
    """

    # TODO implement and improve current_plotting functions
    # TODO Include inverse scaling for features when bo.scaler is not None
        #Partially done for some methods

    save_dir = f"figures/{bo._observer}"
    os.makedirs(save_dir, exist_ok=True)

    common_args = {
        'model': bo._models[OBJECTIVE],
        'initial_data': initial_data,
        'search_space': bo._search_space,
        'scaler': bo._scaler,
        'test_data': test_data,
        'sampled_data': sampled_data,
        'plot_ground_truth': plot_ground_truth,
        'ground_truth_function': ground_truth_function,
        'save_dir': save_dir,
        'count': count
    }
    plot_functions = {
        'log_reg': plot_model,
        'circle': plot_circle,
        'nerve_block': plot_nerve_block,
        'rosenbruck': plot_surface,
        'axon_single': plot_surface,
        'axon_double': plot_surface,
        'axon_threshold': plot_surface,
        'vlmop2': plot_pareto_2d,
        'multiobjective': plot_pareto_2d
    }
    if bo._observer in plot_functions:
        plot_func = plot_functions[bo._observer]
        if bo._observer in ['rosenbruck', 'axon_single', 'axon_double', 'axon_threshold']:
            plot_func(*args, **common_args, **vargs)
        else:
            plot_func(*args, **common_args)
    else:
        msg = f"Plot function not implemented for {bo._observer} problems."
        warnings.warn(msg)
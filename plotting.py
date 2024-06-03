from matplotlib.colors import LinearSegmentedColormap
import warnings
import tensorflow as tf
from trieste.observer import OBJECTIVE
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import gpflow
from matplotlib import cm
import os
from trieste.experimental.plotting import plot_bo_points, plot_function_2d
import matplotlib.pyplot as plt
import seaborn as sns
from trieste.models.interfaces import TrainableModelStack
import functools

from trieste.acquisition.multi_objective.pareto import (
    Pareto,
    get_reference_point,
)


plt.style.use('ggplot')

# TODO HV calculation so that previously calculated values are cached
def custom_cmap():
    colors = ['C1', 'C0']
    cmap_name = 'custom_cmap'
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=2)
    return cmap


def plot_model(model, initial_data, search_space=None, test_data=None,
               sampled_data=None, ground_truth=None, init_fun=None,
               save_dir=None,
               count=0):
    """
    Plot a two feature 1 output model
    @param model: Trieste model
    @param initial_data: Trieste Dataset
    @param search_space: Trieste search space
    @param test_data: Data points to evaluate (normally test)
    @param sampled_data: Data points to evaluate (normally the incremental train)
    @param ground_truth: boolean to decide whether ground_truth is plotted
    @param init_fun: the generator function of the problem (available for some toy problems)
    @param save_dir: path to save figure
    @param count: numeric value to include in the figure name
    @return:
    """
    if ground_truth:
        assert init_fun is not None
        assert search_space is not None

    plt.figure(figsize=(6, 3))
    plt.plot(initial_data[0][:, 0],
             initial_data[1][:, 0],
             'ro', mew=2,
             label='Initial samples')

    if ground_truth:
        x_min = search_space._lower.numpy()
        x_max = search_space._upper.numpy()
        x = np.linspace(x_min, x_max, num=50)
        mean = init_fun(x, noise=0)
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


def plot_circle(model, initial_data, search_space=None, test_data=None,
                sampled_data=None, ground_truth=None, init_fun=None, save_dir=None, count=0):
    if ground_truth:
        assert callable(init_fun)
        assert search_space is not None
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    if search_space:
        x_min, y_min = search_space._lower.numpy()
        x_max, y_max = search_space._upper.numpy()
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))

        Z, Zvar = model.predict_y(np.c_[xx.ravel(), yy.ravel()])
        Z = gpflow.likelihoods.Bernoulli().invlink(Z).numpy().squeeze()
        Z = Z.reshape(xx.shape)
        ax.plot_surface(xx, yy, Z, cmap=cm.coolwarm)

    x = initial_data[0].numpy()
    locs = (initial_data[1] == 1).numpy().squeeze()

    ax.scatter(x[locs, 0], x[locs, 1], marker='o', c='k', label='Train-pos')
    ax.scatter(x[~locs, 0], x[~locs, 1], marker='x', c='k', label='Train-neg')

    if ground_truth:
        x_min, y_min = search_space._lower.numpy()
        x_max, y_max = search_space._upper.numpy()
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))

        Z = model.predict_y(np.c_[xx.ravel(), yy.ravel()])
        Z = tf.math.round(Z[0]).numpy().squeeze()
        Z = Z.reshape(xx.shape)
        ax.contour(xx, yy, Z, colors='k', linestyles='solid', linewidths=1)

    if sampled_data:
        ax.scatter(sampled_data[0][:, 0], sampled_data[0][:, 1],
                   marker='x', c='r', label='Sampled datapoints')

    if test_data:
        mean, var = model.predict_y(test_data[0])
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


def plot_surface(model, initial_data, search_space=None, test_data=None,
                 sampled_data=None, ground_truth=None, init_fun=None,
                 feasible_region=None, save_dir=None,
                 count=0):
    """

    @param model: Model from Trieste. TODO in some cases this may be a list with single output.
    @param initial_data:
    @param search_space:
    @param test_data:
    @param sampled_data:
    @param ground_truth:
    @param init_fun:
    @param feasible_region:
    @param save_dir:
    @param count:
    @return:
    """
    if ground_truth:
        assert callable(init_fun)
        assert search_space is not None

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax = [ax1]

    if search_space:
        x_min, y_min = search_space._lower.numpy()
        x_max, y_max = search_space._upper.numpy()
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))
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

def plot_pareto_2d(model, initial_data, search_space=None, test_data=None,
                   sampled_data=None, ground_truth=None, init_fun=None,
                   save_dir=None,
                   count=0):

    if ground_truth:
        assert callable(init_fun)
        assert search_space is not None

    fig, ax = plt.subplots(figsize=(10, 10), ncols=2, nrows=2)
    ax = ax.ravel()

    if search_space:
        x_min, y_min = search_space._lower.numpy()
        x_max, y_max = search_space._upper.numpy()
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))
        #if isinstance(model, TrainableModelStack):
        Z, Zvar = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.numpy()
        Zvar = Zvar.numpy()
        #else:
        #    raise Exception("Case not defined (If Not TrainableModelStack")

        ax[0].contour(xx, yy, Z[:, 0].reshape(*xx.shape), 80, alpha=1)
        ax[1].contour(xx, yy, Z[:, 1].reshape(*xx.shape), 80, alpha=1)

    ax[0].scatter(initial_data[0][:, 0], initial_data[0][:, 1],
                  initial_data[1][:, 0], marker='o', c='C0', label='Initial samples')

    if sampled_data:
        ax[0].scatter(sampled_data[0][:, 0], sampled_data[0][:, 1], marker='o', c='C1', label='Sampled-data')

    if test_data:
        #if isinstance(TrainableModelStack):
        Z, Zvar = model.predict(test_data[0])
        Z = Z.numpy()
        Zvar = Zvar.numpy()
        #else:
        #    raise Exception("Case not defined (If Not TrainableModelStack")
        ax[0].scatter(test_data[0][:, 0], test_data[0][:, 1], Z[0], c='k', marker='x', label='Test samples')
        ax[1].scatter(test_data[0][:, 0], test_data[0][:, 1], Z[0], c='k', marker='x', label='Test samples')

    #Cached value (to avoid modifying it between repeated calls, although it shouldn't change
    # A bit arbitrary how trieste select the ref point.
    # Plotting the pareto front

    Z_init, Zvar_init = model.predict(initial_data[0])
    Z_samp, Zvar_init = model.predict(sampled_data[0])

    Z_init = Z_init.numpy()
    Z_samp = Z_samp.numpy()

    Z_stack = np.vstack([Z_init, Z_samp])

    front = Pareto(Z_stack).front
    ax[2].scatter(Z_init[:, 0], Z_init[:, 1], c='C0', marker='o', label='Initial \n sample')
    ax[2].scatter(Z_samp[:, 0], Z_samp[:, 1], c='purple', marker='o', label='Sampled-data')
    ax[2].scatter(front[:,0], front[:,1], c='r', marker='x', label="Pareto front")

    ref_point = calculate_reference_point(initial_data[1].numpy())

    _idxs = np.arange(1, len(Z_stack) + 1)
    log_vol = [log_hv(Z_stack[:i, :], ref_point) for i in _idxs]
    ax[3].plot(_idxs, log_vol, color='C1')
    ax[3].axvline(x=len(initial_data[0]), color='C2')
    ax[3].grid()

    ax[0].set_title("Objective 1")
    ax[1].set_title("Objective 2")
    ax[2].set_title("Pareto front")

    fig.tight_layout()

    if save_dir is None:
        plt.savefig(f'./figures/plot_{count:02}.png')
    else:
        plt.savefig(f'{save_dir}/plot_{count:02}.png')
    plt.close()
def update_plot(bo, initial_data=None, sampled_data=None, test_data=None, ground_truth=None, init_fun=None,
                save_dir=None, count=0):
    # TODO implement and improve current_plotting functions
    # TODO os.makedirs if save_dir is not None can be run at the beginning here
    if save_dir is None:
        save_dir = "figures/default"
    else:
        save_dir = f"figures/{bo._observer}"

    os.makedirs(save_dir, exist_ok=True)
    if bo._observer == 'log_reg':
        plot_model(model=bo._models[OBJECTIVE], search_space=bo._search_space,
                   initial_data=initial_data, test_data=test_data,
                   sampled_data=sampled_data, ground_truth=ground_truth,
                   init_fun=init_fun, save_dir=save_dir, count=count)
    elif bo._observer == 'circle':
        plot_circle(model=bo._models[OBJECTIVE], search_space=bo._search_space,
                    initial_data=initial_data, test_data=test_data,
                    sampled_data=sampled_data, ground_truth=ground_truth,
                    init_fun=init_fun, save_dir=save_dir, count=count)
    elif bo._observer == 'rosenbruck':
        plot_surface(model=bo._models[OBJECTIVE], search_space=bo._search_space,
                     initial_data=initial_data, test_data=test_data,
                     sampled_data=sampled_data, ground_truth=ground_truth,
                     init_fun=init_fun, save_dir=save_dir, count=count)
    elif bo._observer in ['axon_single', 'axon_double', 'axon_threshold']:
        plot_surface(model=bo._models[OBJECTIVE], search_space=bo._search_space,
                     initial_data=initial_data, test_data=test_data,
                     sampled_data=sampled_data, ground_truth=ground_truth,
                     init_fun=init_fun, save_dir=save_dir, count=count)
    elif bo._observer in ['vlmop2','multiobjective']:
        plot_pareto_2d(model=bo._models[OBJECTIVE], search_space=bo._search_space,
                       initial_data=initial_data, test_data=test_data,
                       sampled_data=sampled_data, ground_truth=ground_truth,
                       init_fun=init_fun, save_dir=save_dir, count=count)
    else:
        msg = f"Plot fun Not implemented for {bo._observer} problems."
        warnings.warn(msg)
        pass

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
plt.style.use('ggplot')

def custom_cmap():
    colors = ['C1', 'C0']
    cmap_name = 'custom_cmap'
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=2)
    return cmap


def plot_model(model, train_data, search_space=None, eval_points=None,
               sampled_data=None, ground_truth=None, init_fun=None,
               save_dir = None,
               count=0):
    if ground_truth:
        assert init_fun is not None
        assert search_space is not None

    plt.figure(figsize=(6, 3))
    plt.plot(train_data[0][:, 0],
             train_data[1][:, 0],
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

    if eval_points:
        mean, var = model.predict_y(eval_points[0])
        plt.plot(eval_points[0][:, 0], mean, 'C0', lw=2, label='Prediction')
        plt.fill_between(eval_points[0][:, 0],
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

def plot_circle(model, train_data, search_space=None, eval_points=None,
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

    x = train_data[0].numpy()
    locs = (train_data[1] == 1).numpy().squeeze()

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

    if eval_points:
        mean, var = model.predict_y(eval_points[0])
        preds = tf.math.round(mean).numpy()
        corr_bool = (preds > 0).squeeze()

        ax.scatter(eval_points[0].numpy()[corr_bool, 0], eval_points[0].numpy()[corr_bool, 1], c='C0', marker='o',
                   label='Pos-test')
        ax.scatter(eval_points[0].numpy()[~corr_bool, 0], eval_points[0].numpy()[~corr_bool, 1], c='C1', marker='x',
                   label='Neg-test')

    ax.legend()
    #ax.set_xlim(-1, 1)
    #ax.set_ylim(-1, 1)
    if save_dir is None:
        plt.savefig(f'./figures/plot_{count:02}.png')
    else:
        plt.savefig(f'{save_dir}/plot_{count:02}.png')
    plt.close()

def plot_surface(model, train_data, search_space=None, eval_points=None,
                sampled_data=None, ground_truth=None, init_fun=None,
                 feasible_region=None, save_dir=None,
                 count=0):

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
        #ax[1].contourf(xx, yy, Zvar, alpha=0.8)

    if train_data is not None:
        ax[0].scatter(train_data[0][:, 0], train_data[0][:, 1],
                   train_data[1][:, 0], marker='o', c='k', label='Train-pos')

    if feasible_region:
        # TODO
        pass

    if sampled_data:
        y_samps, _ = model.predict_y(sampled_data[0])
        ax[0].scatter(sampled_data[0][:, 0], sampled_data[0][:, 1],
                   y_samps, marker='x', c='r', label='Sampled-Points')

    if eval_points:
        mean, var = model.predict_y(eval_points[0])
        preds = tf.math.round(mean).numpy()

        ax[0].scatter(eval_points[0][:, 0], eval_points[0][:, 1], Z,  c='k', marker='o', label='Samples added')

    ax[0].legend()
    if save_dir is None:
        plt.savefig(f'./figures/plot_{count:02}.png')
    else:
        plt.savefig(f'{save_dir}/plot_{count:02}.png')
    plt.close()


def update_plot(bo,
                eval_points=None,
                train_data=None,
                sampled_data=None,
                ground_truth=None,
                init_fun=None,
                save_dir= None,
                count=0):
    # TODO implement and improve current_plotting functions
    # TODO os.makedirs if save_dir is not None can be run at the beginning here
    if save_dir is None:
        save_dir = "figures/default"
    else:
        save_dir = f"figures/{bo._observer}"

    os.makedirs(save_dir, exist_ok=True)
    if bo._observer == 'log_reg':
        plot_model(model=bo._models[OBJECTIVE],  search_space=bo._search_space,
                   train_data=train_data, eval_points=eval_points,
                   sampled_data=sampled_data, ground_truth=ground_truth,
                   init_fun=init_fun, save_dir=save_dir, count=count)
    elif bo._observer == 'circle':
        plot_circle(model=bo._models[OBJECTIVE],  search_space=bo._search_space,
                   train_data=train_data, eval_points=eval_points,
                   sampled_data=sampled_data, ground_truth=ground_truth,
                   init_fun=init_fun, save_dir=save_dir, count=count)
    elif bo._observer == 'rosenbruck':
        plot_surface(model=bo._models[OBJECTIVE], search_space=bo._search_space,
                   train_data=train_data, eval_points=eval_points,
                   sampled_data=sampled_data, ground_truth=ground_truth,
                   init_fun=init_fun, save_dir=save_dir, count=count)
    elif bo._observer == 'AxonSim':
        plot_surface(model=bo._models[OBJECTIVE], search_space=bo._search_space,
                   train_data=train_data, eval_points=eval_points,
                   sampled_data=sampled_data, ground_truth=ground_truth,
                   init_fun=init_fun, save_dir=save_dir, count=count)
    else:
        msg = f"Plot fun Not implemented for {bo._observer} problems."
        warnings.warn(msg)
        pass
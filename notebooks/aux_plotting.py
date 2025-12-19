import warnings

import matplotlib.pyplot as plt
import numpy as np

DPI = 300


def custom_scatter(
    x_values,
    y_values,
    labels=[],
    xlabel="I",
    ylabel="Pulse duration",
    title="Initial search space",
    figsize=(5, 3),
    dpi=None,
    save_fig=True,
    save_dir=None,
):

    if len(labels) == 0:
        labels = [""] * len(x_values)

    if dpi is None:
        dpi = DPI

    assert (
        len(x_values) == len(y_values) == len(labels)
    ), "x_values, y_values and labels must have the same length"

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for x, y, label in zip(x_values, y_values, labels):
        ax.scatter(x, y, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid()
    ax.legend()
    if save_fig and save_dir:
        plt.savefig(save_dir, format="svg")


def plot_uncertainty(
    bo,
    scaler,
    figure_path,
    x_axis_label=None,
    y_axis_label=None,
    grid_size=30,
    cross_pair=None,
    init_df=None,
    collected_df=None,
    train_ix=None,
    test_ix=None,
    s=10,
    dpi=300,
    figsize=(6, 6),
):
    """Plot uncertainty heatmap with search space exploration.

    Args:
        bo (_type_): _description_
        scaler (_type_): _description_
        figure_path (_type_): _description_
        init_df (_type_, optional): _description_. Defaults to None.
        train_ix (_type_, optional): _description_. Defaults to None.
        test_ix (_type_, optional): _description_. Defaults to None.
    """

    # Extract original feature space bounds
    x_min, y_min = scaler.feature_min
    x_max, y_max = scaler.feature_max

    x_vals = np.linspace(x_min, x_max, grid_size)
    y_vals = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x_vals, y_vals)

    test_grid_out = np.column_stack([X.ravel(), Y.ravel()])
    test_grid_out = scaler.transform(test_grid_out)

    grid_means, grid_vars = bo.models["OBJECTIVE"].predict(test_grid_out)
    grid_means = grid_means.numpy()
    grid_vars = grid_vars.numpy()

    # Compute per-sample uncertainty and reshape
    per_sample_uncertainty = np.mean(grid_vars, axis=1)
    uncertainty_grid = per_sample_uncertainty.reshape(grid_size, grid_size)

    if x_axis_label is not None and y_axis_label is not None and init_df is not None:
        x_plot_col = init_df.columns[0]
        y_plot_col = init_df.columns[1]

    elif init_df is not None:
        x_axis_label = init_df.columns[0]
        y_axis_label = init_df.columns[1]
        x_plot_col = init_df.columns[0]
        y_plot_col = init_df.columns[1]
    else:
        warnings.warn("No labels provided and init_df is None. Using default labels.")
        x_axis_label = "Feature 1"
        y_axis_label = "Feature 2"

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(
        uncertainty_grid,
        cmap="inferno",
        aspect="auto",
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
    )  # Ensures alignment with original scale
    plt.colorbar(im, ax=ax, label="Uncertainty")

    if init_df is not None:
        # Overlay scatter plot
        if train_ix is not None:
            ax.scatter(
                init_df[x_plot_col].values[train_ix],
                init_df[y_plot_col].values[train_ix],
                s=s,
                label="Train samples",
                color="blue",
                edgecolors="black",
            )
        if test_ix is not None:
            ax.scatter(
                init_df[x_plot_col].values[test_ix],
                init_df[y_plot_col].values[test_ix],
                s=s,
                label="Hold-out samples",
                color="red",
                edgecolors="black",
            )

    if cross_pair is not None:
        ax.scatter(
            cross_pair[:, 0],
            cross_pair[:, 1],
            s=s,
            label="Next sample",
            color="k",
            marker="x",
        )

    if collected_df is not None:
        ax.scatter(
            collected_df[:, 0],
            collected_df[:, 1],
            s=s,
            label="Collected samples",
            color="green",
            edgecolors="black",
        )

    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)
    ax.set_title("Uncertainty Heatmap with Search Space Exploration")

    # Set tick positions and labels
    tick_positions = np.linspace(x_min, x_max, 5)  # Adjust as needed
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(np.round(tick_positions, 2))

    tick_positions = np.linspace(y_min, y_max, 5)  # Adjust as needed
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(np.round(tick_positions, 2))

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    # ax.grid()
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25)  # Increase if legend is clipped

    plt.savefig(figure_path / "uncertainty_and_sampling.svg", format="svg")
    plt.show()

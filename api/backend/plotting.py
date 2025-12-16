# api/backend/plotting.py
import warnings

import numpy as np
import requests

from ..frontend.components.config_forms import plot_configurations

# TODO, plotting functions for Pareto
# TODO, plotting functions for 3D search shapes (see Notebook in NeuroAAA repo)
# TODO make GRID_POINTS a configurable parameter elsewhere
GRID_POINTS = 20


def generate_line_plot(context: dict, params: dict, with_meta: bool) -> dict | None:
    """Generates a data packet for a multi-line plot against a common x-axis.

    Required context:
        - 'lines_dict': The dictionary with the data.

    Required params:
        - 'x_col': Name of the column for the x-axis.
        - 'y_cols': A list of column names for the y-axis series.

    Optional params:
        - 'y_labels': A list of labels for the legend, corresponding to y_cols.
                      If not provided, y_cols will be used as labels.
        - 'title', 'xaxis_title', 'yaxis_title': Strings for plot and axis titles.
    """

    dict_data = context.get("lines_dict")
    if dict_data is None:
        warnings.warn("lines_dict not found in context.")
        return None

    x_col = params["x_col"]
    y_cols = params["y_cols"]
    y_labels = params.get(
        "y_labels", y_cols
    )  # Fallback to column names if labels not provided

    if len(y_cols) != len(y_labels):
        raise ValueError("The number of 'y_cols' must match the number of 'y_labels'.")

    required_cols = [x_col] + y_cols
    if not all(col in dict_data for col in required_cols):
        print(f"Warning: One or more columns missing for line plot: {required_cols}")
        return None

    # Get the x-axis data once
    x_data = dict_data[x_col]
    x_data = x_data.tolist() if hasattr(x_data, "tolist") else list(x_data)

    # Create a list of traces, one for each line
    traces = []
    for y_col, label in zip(y_cols, y_labels):
        trace = {
            "x": x_data,
            "y": (
                dict_data[y_col].tolist()
                if hasattr(dict_data[y_col], "tolist")
                else list(dict_data[y_col])
            ),
            "name": label,  # Legend
            "type": "scatter",
            "mode": "lines",
        }
        traces.append(trace)

    # The 'data' key now contains a list of trace objects
    return_json = {"data": traces, "meta": None}
    if with_meta:
        return_json["meta"] = {
            "title": params.get("title", "Line Plot"),
            "xaxis_title": params.get("xaxis_title", x_col),
            "yaxis_title": params.get("yaxis_title", "Value"),
        }

    return return_json


def generate_scatter_from_dataframe(
    context: dict, params: dict, with_meta: bool
) -> dict | None:
    """Generates a data packet for a scatter plot from a pandas DataFrame.

    Required context:
        - 'results_df': The pandas DataFrame with all experiment results.

    Required params:
        - 'x_col': Name of the column for the x-axis.
        - 'y_col': Name of the column for the y-axis.
        - 'color_col': Name of the column for the color/value of the points.
    Optional params:
        - 'title', 'xaxis_title', 'yaxis_title': Strings for plot and axis titles.
    """
    df = context.get("results_df")
    if df is None:
        warnings.warn("results_df not found in context.")
        return None

    x_col, y_col, color_col = params["x_col"], params["y_col"], params["color_col"]

    if not all(col in df.columns for col in [x_col, y_col, color_col]):
        print(
            f"Warning: One or more columns missing for scatter plot: {x_col}, {y_col}, {color_col}"
        )
        return None  # Or return an empty packet

    feature_names = [x_col, y_col]
    observation_names = [color_col]

    return_json = {
        "data": {
            "x": df[x_col].tolist(),
            "y": df[y_col].tolist(),
            "class": df[color_col].tolist(),
        },
        "meta": None,
    }
    if with_meta:
        return_json["meta"] = {
            "feature_names": feature_names,
            "observation_names": observation_names,
            "xaxis_title": params.get(x_col, None),
            "yaxis_title": params.get(y_col, None),
        }
    return return_json


def generate_contour_from_cajal_results(
    context: dict, params: dict, with_meta: bool
) -> dict | None:
    """Generates a data packet for a contour plot from an array of values

    Required context:
        - 'full_observations': Array con full (pre-processed response). Currently used for
                            neural responses over time/nodes.

    Required params:
        - 'x_col', 'y_col': The names of the features for the axes.
        - 'z_source': Either 'mean' or 'variance'.
    Optional params:
        - 'title', 'xaxis_title', 'yaxis_title': Strings for plot and axis titles.
        - 'zmin', 'zmax' the ranges to scale the colorbar.

    """
    full_obs = context.get("full_observations")
    if full_obs is None:
        warnings.warn("full_observations not found in context.")
        return None

    voltage_matrix = np.array(full_obs["voltage"])
    # Y-axis: Nodes (Columns of voltage matrix)
    y_index = np.arange(1, voltage_matrix.shape[1] + 1)
    # X-axis: Time
    x_index = np.array(full_obs["time"])
    z_values = voltage_matrix

    return_json = {
        "data": {"x": x_index.tolist(), "y": y_index.tolist(), "z": z_values.tolist()},
        "meta": None,
    }

    if with_meta:
        return_json["meta"] = {
            "grid_size": GRID_POINTS,
            "feature_names": ["Time (ms)", "Node index"],
            "title": params.get("title", "AP propagation"),
            "zmin": params.get("zmin", None),
            "zmax": params.get("zmax", None),
            "xticks": x_index.tolist()[::10],
            "yticks": y_index.tolist()[::10],
            "xaxis_title": params.get("Time (ms)", None),
            "yaxis_title": params.get("Node index", None),
        }
    return return_json


def generate_contour_from_model(
    context: dict, params: dict, with_meta: bool
) -> dict | None:
    """
    Generate a data packet for a contour plot from a predictive model.

    Required context:
        - 'bo_model': The predictive model object (e.g., from a BO library).
        - 'scaler': A scaler object with feature_min/max and output_min/max attributes.

    Required params:
        - 'x_col', 'y_col': The names of the features for the axes.
        - 'z_source': Either 'mean' or 'variance'.
    Optional params:
        - 'title', 'xaxis_title', 'yaxis_title': Strings for plot and axis titles.
        - 'zmin', 'zmax' the ranges to scale the colorbar.
    """
    model = context["bo_model"]
    scaler = context["scaler"]
    x_col, y_col, z_source = params["x_col"], params["y_col"], params["z_source"]

    # Create the grid in original feature space
    x_orig = np.linspace(scaler.feature_min[0], scaler.feature_max[0], GRID_POINTS)
    y_orig = np.linspace(scaler.feature_min[1], scaler.feature_max[1], GRID_POINTS)

    # Create the grid in scaled model input space
    x_scaled = np.linspace(scaler.output_min[0], scaler.output_max[0], GRID_POINTS)
    y_scaled = np.linspace(scaler.output_min[1], scaler.output_max[1], GRID_POINTS)
    mx_scaled, my_scaled = np.meshgrid(x_scaled, y_scaled, indexing="xy")

    model_inputs = np.column_stack([mx_scaled.ravel(), my_scaled.ravel()])

    # Predict based on the z_source parameter
    if z_source == "mean":
        mean, _ = model.models["OBJECTIVE"].predict_y(model_inputs)
        z_values = mean.numpy().reshape(mx_scaled.shape)

    elif z_source == "variance":
        _, variance = model.models["OBJECTIVE"].predict_y(model_inputs)
        z_values = variance.numpy().reshape(mx_scaled.shape)

    else:
        raise ValueError(
            f"Unknown z_source: '{z_source}'. Must be 'mean' or 'variance'."
        )

    return_json = {
        "data": {"x": x_orig.tolist(), "y": y_orig.tolist(), "z": z_values.tolist()},
        "meta": None,
    }

    if with_meta:
        return_json["meta"] = {
            "grid_size": GRID_POINTS,
            "feature_names": [x_col, y_col],
            "title": params.get("title", "Contour Plot"),
            "zmin": params.get("zmin", None),
            "zmax": params.get("zmax", None),
            "xticks": x_orig.tolist(),
            "yticks": y_orig.tolist(),
            "xaxis_title": params.get(x_col, None),
            "yaxis_title": params.get(y_col, None),
        }
    return return_json


def prepare_data_bundle(
    experiment_name: str, context: dict, with_meta: bool = False
) -> dict[str, dict]:
    """
    Prepare the full data_sources bundle for a given experiment update.

    Args:
        experiment_name: The name of the experiment (e.g., 'cajal_ap_block').
        context: A dictionary containing all raw data needed by the generators.
                 Example: {'results_df': df,
                 'lines_dict': dict,
                 'bo_model': model, 'scaler': scaler,
                 'full_observations': full_observations}

    Returns:
        A dictionary of data packets, ready to be sent to the Flask app.
    """
    if experiment_name not in plot_configurations:
        warnings.warn(f"No plot configuration found for experiment '{experiment_name}'")
        return {}

    configs = plot_configurations[experiment_name]
    data_sources = {}

    for config in configs:
        src = config.get("src")
        if src is None:
            warnings.warn(
                f"src cannot be None for experiment {experiment_name}  and config: {config}"
            )
            continue
        if not src or src in data_sources:
            continue

        generator_name = config.get("generator")
        if not generator_name or generator_name not in DATA_GENERATORS:
            warnings.warn(
                f"No generator '{generator_name}' found for src '{src}'. Skipping."
            )
            continue

        params = config.get("params", None)

        if not params:
            warnings.warn(
                f"No params for '{generator_name}' found for src '{src}'. Skipping."
            )
            continue

        assert isinstance(params, dict), "params is not a dictionary"

        # Get the generator function from the registry
        generator_func = DATA_GENERATORS[generator_name]

        # Call the generator and store the result
        try:
            data_packet = generator_func(context, params, with_meta)
            if data_packet:
                print(
                    f"For src {src}, following keys are present: \n {data_packet.keys()}"
                )
                data_sources[src] = data_packet

        except Exception as e:
            warnings.warn(
                f"Error generating data for src '{src}', func: {generator_name}: {e}"
            )

    return data_sources


DATA_GENERATORS = {
    "scatter_from_dataframe": generate_scatter_from_dataframe,
    "contour_from_model": generate_contour_from_model,
    "plot_line": generate_line_plot,
    "contour_from_cajal_results": generate_contour_from_cajal_results,
    # Add new generator mappings here as you create them
}


def send_plot_bundle(
    post_url: str, experiment_name: str, data_sources: dict[str, dict]
) -> None:
    """
    Send a batch update containing experiment results or status information
    to the Flask frontend.

    This function performs a POST request to the Flask endpoint
    ``/update_plots_bundle``. If ``post_url``is empty the function exits
    silently.

    Parameters
    ----------
    post_url : str
        Base URL of the Flask server (e.g., ``"http://localhost:5000"``).
        If empty, no request is sent.

    experiment_name : str
        Name of the experiment that the UI should associate the update with.

    data_sources : dict[str, dict]
        A dictionary containing structured data to send to the UI.
        Keys usually correspond to plot names, log entries, or status fields.

        Example:
            {
                "acquisition": {...},
                "model_metrics": {...},
                "optimizer_state": {...}
            }

    Returns
    -------
    None

    Notes
    -----
    - If the request fails, a warning is issued rather than raising an error.
    """
    if not post_url:
        return

    json_payload = {"experiment": experiment_name, "sources": data_sources}
    try:
        response = requests.post(f"{post_url}/update_plots_bundle", json=json_payload)
        response.raise_for_status()  # Raises an exception for bad status codes
    except requests.exceptions.RequestException as e:
        msg = f"Error updating Flask UI: {e}"
        warnings.warn(msg)

# api/backend/parameters.py
from flask import Response, jsonify

from api.frontend.components.config_forms import (
    config_model,
    config_optimizer,
    config_problem,
    matlab_experiments,
    python_experiments,
)


def get_parameters_logic(data: dict) -> tuple[Response, int]:
    """
    Auxiliar function for the get_parameters endpoint.
    Validate the incoming request data and return configuration parameters.

    This auxiliar function inspects the "type" field in the provided
    dictionary and dispatches to the appropriate configuration loader:

    - "problem_parameters": Loads experiment-specific parameter templates.
      Requires an "experiment" field that must map to known MATLAB or Python
      experiments.

    - "model": Loads configuration for the requested model. Requires a "model" field.

    - "optimizer": Returns the common optimizer configuration.

    Parameters
    ----------
    data : dict
        Request payload containing the configuration query. Must include:
        - "type": One of {"problem_parameters", "model", "optimizer"}.
        - Additional required fields depending on the type.

    Returns
    -------
    tuple[Response, int]
        A Flask JSON response with:
        - 200 on success, containing the resolved configuration.
        - 400 if required fields are missing or invalid.

    Raises
    ------
    NotImplementedError
        If an unsupported parameter type is provided.
    """

    if not data:
        return jsonify({"error": "No data provided"}), 400

    parameters_type = data.get("type")
    if not parameters_type:
        return jsonify({"error": "No parameters' type provided"}), 400

    if parameters_type == "problem_parameters":
        experiment = data.get("experiment")
        if not experiment:
            return jsonify({"error": "No experiment provided"}), 400

        if experiment in matlab_experiments:
            params = config_problem(matlab_experiments[experiment])
        elif experiment in python_experiments:
            params = config_problem(python_experiments[experiment])
        else:
            return jsonify({"error": f"Experiment {experiment} not found"}), 400

    elif parameters_type == "model":
        model = data.get("model")
        if not model:
            return jsonify({"error": "No model provided"}), 400
        params = config_model(model)

    elif parameters_type == "optimizer":
        # optimizer = data.get("optimizer")
        params = config_optimizer()
    else:
        raise NotImplementedError(f"parameter_type {parameters_type} not supported")

    return jsonify(params), 200

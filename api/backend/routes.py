# api/backend/routes.py

import json
import subprocess
import threading

from flask import Response, jsonify, render_template, request
from jinja2 import TemplateNotFound

from api.app import DEFAULT_PORT
from api.backend.parameters import get_parameters_logic
from api.backend.utils.general import close_flush_files
from api.backend.utils.plot_helpers import emit_plot_bundle
from api.frontend.components.config_forms import (
    acquisition_map,
    config_function,
    experiments_types,
    matlab_experiments,
    model_map,
    plot_configurations,
    python_experiments,
)

from .process_routines import monitor_process, prepare_experiment, stop_process

experiment_list = list(matlab_experiments.keys()) + list(python_experiments.keys())
model_list = list(model_map.keys())
acquisition_list = list(acquisition_map.keys())

process = None
process_lock = threading.Lock()  # Lock to ensure thread safety


def register_routes(app, socketio):

    @app.route("/", methods=["GET"])
    def index() -> str:
        """
        Render the main frontend page.

        The template 'frontend.html' is rendered with the server port passed
        as a template variable.

        Returns
        -------
        str
            The rendered HTML page.
        """
        port = app.config.get("PORT", DEFAULT_PORT)  # Default to 9000 if not set
        return render_template("frontend.html", port=port)

    @app.route("/experiments", methods=["GET"])
    def get_experiments() -> Response:
        """
        Return the list of all experiments in JSON format.

        This endpoint responds to GET requests and returns the current
        ``experiment_list`` as a JSON array.

        Returns
        -------
        flask.Response
            A JSON response containing the list of experiments.
        """
        return jsonify(experiment_list)

    @app.route("/models", methods=["GET"])
    def get_models() -> Response:
        """
        Return the list of available models in JSON format.

        This endpoint responds to GET requests and returns the current
        ``model_list`` as a JSON array.

        Returns
        -------
        flask.Response
            A JSON response containing the list of models.
        """
        return jsonify(model_list)

    @app.route("/acquisitions", methods=["GET"])
    def get_acquisitions() -> Response:
        """
        Return the list of available acquisitions in JSON format.

        This endpoint responds to GET requests and returns the current
        ``acquisition_list`` as a JSON array.

        Returns
        -------
        flask.Response
            A JSON response containing the list of acquisitions.
        """
        return jsonify(acquisition_list)

    @app.route("/fun_parameters", methods=["POST"])
    def get_fun_parameters() -> Response | tuple[Response, int]:
        """
        Return the list of available acquisitions in JSON format.

        This endpoint accepts a POST request with JSON data containing the
        ``function`` key. It calls ``config_function`` to retrieve the
        function's configuration and returns it as JSON. If the input is missing
        or an error occurs during processing, an error message is returned with
        HTTP status 400.

        Parameters
        ----------
        function : str
            Name of the function for which to retrieve configuration parameters.
            Provided in the JSON payload as {"function": "<function_name>"}.

        Returns
        -------
        flask.Response
            A JSON response containing the function configuration, or an error
            message with status 400 if the input is invalid or an exception occurs.
        """
        data = request.json
        if data is None:
            return jsonify({"error": "No data provided"}), 400

        function = data.get("function")

        if not function:
            return jsonify({"error": "No experiment provided"}), 400
        try:
            fun_config = config_function(function)
            return jsonify(fun_config)
        except Exception as e:
            msg = str(e)
            return jsonify({"error": msg}), 400

    @app.route("/parameters", methods=["POST"])
    def get_parameters() -> Response | tuple[Response, int]:
        """
        Return configuration parameters based on the requested type.

        This endpoint accepts a POST request with JSON data specifying the
        ``type`` of parameters to retrieve. Supported types include:

        - ``"problem_parameters"``: requires an ``experiment`` key; returns problem parameters
        for the specified experiment.
        - ``"model"``: requires a ``model`` key; returns model configuration parameters.
        - ``"optimizer"``: returns optimizer configuration parameters.

        If required fields are missing or an unsupported type is requested,
        an error message is returned with status 400.

        Parameters
        ----------
        type : str
            The type of parameters requested. Must be one of
            ``"problem_parameters"``, ``"model"``, or ``"optimizer"``.
        experiment : str, optional
            Name of the experiment, required if ``type`` is ``"problem_parameters"``.
        model : str, optional
            Name of the model, required if ``type`` is ``"model"``.
        optimizer : str, optional (TODO not implemented yet, required for extended optimizer config)
            Name of the optimizer, required if ``type`` is ``"optimizer"``.

        Returns
        -------
        flask.Response | tuple
            A JSON response containing the requested parameters, or a JSON error
            message with HTTP status 400 if input is invalid.
        """
        data = request.json or {}
        try:
            response = get_parameters_logic(data)
            return response
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": "Internal error", "details": str(e)}), 500

    @app.route("/experiment_type", methods=["POST"])
    def get_experiment_type() -> tuple[Response, int]:
        """
        Return the type of a given experiment (e.g. regression or classification)

        This endpoint accepts a POST request with JSON data containing the
        ``experiment`` key. It looks up the experiment in either the Python or
        MATLAB experiments dictionaries and returns its type. If the experiment
        is not provided or unknown, an error message is returned with an
        appropriate HTTP status.

        Parameters
        ----------
        experiment : str
            Name of the experiment to query. Provided in the JSON payload as
            {"experiment": "<experiment_name>"}.

        Returns
        -------
        flask.Response | tuple[flask.Response, int]
            A JSON response containing the experiment type, or an error message
            with HTTP status 400 (missing data) or 404 (unknown experiment).
        """
        data = request.json
        if not data or "experiment" not in data:
            return jsonify({"error": "Experiment name required"}), 400

        sel_experiment = data["experiment"]
        exp_name = python_experiments.get(sel_experiment) or matlab_experiments.get(
            sel_experiment
        )
        if not exp_name:
            return (
                jsonify(
                    {"error": f"No configuration for experiment: {sel_experiment}'"}
                ),
                404,
            )

        if exp_name not in experiments_types:
            return (
                jsonify({"error": f"No type defined for experiment: {exp_name}"}),
                404,
            )

        return jsonify({"type": experiments_types[exp_name]}), 200

    @app.route("/plot_config", methods=["POST"])
    def get_plot_config() -> Response | tuple[Response, int]:
        """
        Return the plot configuration for a specified experiment.

        This endpoint accepts a POST request with JSON data containing the
        ``experiment`` key. It looks up the experiment in either the Python or
        MATLAB experiments dictionaries and returns the corresponding plot
        configuration. If no configuration exists, an empty list is returned
        with a warning. If the experiment is missing or unknown, an error
        message is returned with an appropriate HTTP status.

        Parameters
        ----------
        experiment : str
            Name of the experiment to query. Provided in the JSON payload as
            {"experiment": "<experiment_name>"}.

        Returns
        -------
        flask.Response | tuple[flask.Response, int]
            A JSON response containing the plot configuration, a warning if no
            configuration exists, or an error message with status 400 or 404.
        """
        data = request.json

        if not data or "experiment" not in data:
            return jsonify({"error": "Experiment name required"}), 400

        sel_experiment = data["experiment"]
        exp_name = python_experiments.get(sel_experiment) or matlab_experiments.get(
            sel_experiment
        )
        if not exp_name:
            return (
                jsonify(
                    {"error": f"No configuration for experiment: {sel_experiment}'"}
                ),
                404,
            )

        if exp_name not in plot_configurations:
            return (
                jsonify(
                    {
                        "warning": f"No plot config defined for experiment '{exp_name}'",
                        "config": [],
                    }
                ),
                200,
            )

        return jsonify({"config": plot_configurations[exp_name]})

    @app.route("/plot", methods=["GET"])
    def plot() -> tuple[Response, int] | str:
        """
        Render a plot page based on the specified type and ID.

        This endpoint accepts GET requests with query parameters specifying
        the plot ``id`` and optionally the ``type`` and ``port``. It selects
        the corresponding Jinja template and renders it with the provided
        port and plot ID. If the plot ID is missing or the template does not
        exist, an error message is returned with HTTP status 400.

        Query Parameters
        ----------------
        id : str
            The unique identifier of the plot to render. Required.
        type : str, optional
            The type of plot template to use. Defaults to 'default'.
        port : int, optional
            The port number to pass to the template. Defaults to the
            configured server port.

        Returns
        -------
        tuple[Response, int] | str
            The rendered HTML page as str, or an error message with status 400 if
            the plot ID is missing or the template is not found.
        """
        # Get the port number from query parameters
        port = request.args.get("port", default=DEFAULT_PORT, type=int)

        # Get the plot type from query parameters, defaulting to 'default'
        plot_type = request.args.get("type", default="default", type=str)
        plot_id = request.args.get("id", default=None)
        if plot_id is None:
            return jsonify({"error": "Plot requires an id."}), 400

        template_name = f"plot_{plot_type}.html"
        # Choose the template based on the plot type
        try:
            template = app.jinja_env.get_or_select_template(template_name)
            return template.render(port=port, plot_id=plot_id)
        except TemplateNotFound:
            return jsonify({"error": f"Template '{template_name}' not found."}), 400

    @app.route("/update_plots_bundle", methods=["POST"])
    def update_plots_bundle() -> tuple[Response, int] | tuple[dict, int]:
        """
        Process and emit a bundle of plot data to the configured plots for an experiment.

        This endpoint accepts a POST request containing a JSON payload with an
        ``experiment`` key and a ``sources`` dictionary. It looks up the plot
        configurations for the specified experiment and emits the corresponding
        data to each plot via Socket.IO. Each emitted message contains a standardized
        ``data`` and ``meta`` structure, including the plot ID and type.

        Parameters
        ----------
        experiment : str
            Name of the experiment. Provided in the JSON payload as
            {"experiment": "<experiment_name>"}.
        sources : dict
            Dictionary mapping source keys to their corresponding data packets.
            Each packet can contain ``data`` and optional ``meta`` fields.

        Returns
        -------
        tuple[flask.Response, int] | tuple[dict, int]
            A JSON response confirming successful processing of the data bundle,
            or an error message with status 400 if the input is invalid.
        """
        payload = request.get_json()
        if not payload or "experiment" not in payload or "sources" not in payload:
            return jsonify({"error": "Invalid data bundle format"}), 400

        exp_name = payload["experiment"]
        data_sources = payload["sources"]

        # Find the plot configurations for this experiment
        if exp_name not in plot_configurations:
            return (
                jsonify(
                    {"warning": f"No plot config defined for experiment '{exp_name}'"}
                ),
                200,
            )

        emit_plot_bundle(plot_configurations[exp_name], data_sources, socketio)

        return (
            jsonify(
                {"message": f"Data bundle for '{exp_name}' processed successfully"}
            ),
            200,
        )

    @app.route("/start", methods=["POST"])
    def start_experiment() -> Response | tuple[Response, int]:
        """
        Validate input data and construct/run the experiment subprocess.

        This endpoint accepts a POST request with JSON data specifying the
        experiment, model, and optimizer parameters. For axon-related problems it
        contains also the pulse parameters.
        The input data is validated and converted as necessary, and a subprocess
        is launched to run the experiment. The global ``process`` variable is used to
        track the running experiment, and a background thread monitors its
        completion. If a test command is provided in the payload, it is run
        directly instead of constructing the full experiment command.

        # TODO. Eventually this could be extended to support parallel experiments
        Parameters
        ----------
        experimentParameters : dict
            Dictionary of experiment-specific parameters, including the
            experiment name.
        pulseParameters : dict
            Dictionary of pulse parameters for the experiment.
        modelParameters : dict
            Dictionary of model configuration parameters.
        optimizerParameters : dict
            Dictionary of optimizer configuration parameters.
        test_command : list[str], optional
            A custom command containing a boolean.
            If true, it runs a unit test case.

        Returns
        -------
        tuple[flask.Response, int]
            A JSON response indicating the status of the experiment:
            - ``started``: the process was successfully launched.
            - ``already running``: another experiment process is currently running.
            - error messages with status 400 or 500 if input is invalid or an
            unexpected error occurs.
        """

        global process
        data = request.json

        if data is None:
            return jsonify({"error": "No data provided"}), 400

        if not isinstance(data, dict):
            try:
                data = json.loads(data)
            except Exception as e:
                return jsonify({"error": f"Invalid JSON format: {e}"}), 400

        with process_lock:
            if process is not None:
                return jsonify({"status": "already running", "pid": process.pid}), 200
            try:
                if "test_command" in data:
                    command = data["test_command"]
                else:
                    port = app.config.get("PORT", DEFAULT_PORT)
                    command = prepare_experiment(data, port)

                process = subprocess.Popen(command, start_new_session=True)
                threading.Thread(
                    target=monitor_process, args=(process,), daemon=True
                ).start()

                return jsonify({"status": "started", "command": command}), 200
            except Exception as e:
                return jsonify({"status": "Unexpected error", "error": str(e)}), 500

    @app.route("/status", methods=["GET"])
    def check_experiment() -> Response | tuple[Response, int]:
        """
        Check the current status of the experiment process.

        This endpoint accepts GET requests and returns whether an experiment
        subprocess is currently running. The global ``process`` variable is
        used to track the running experiment, and access is synchronized
        using ``process_lock``. Any unexpected errors are caught and returned
        with status 500.

        Returns
        -------
        flask.Response | tuple[flask.Response, int]
            A JSON response indicating the status:
            - ``running``: the experiment process is active.
            - ``not running``: no experiment process is currently running.
            - error message with status 500 if an unexpected exception occurs.
        """
        global process
        with process_lock:
            proc = process
        try:
            if proc is None or proc.poll() is not None:
                return jsonify({"status": "not running"})
            return jsonify({"status": "running"})
        except Exception as e:
            return jsonify({"status": "Unexpected error", "error": str(e)}), 500

    @app.route("/stop", methods=["POST"])
    def stop_experiment() -> tuple[Response, int]:
        """
        Stop the currently running experiment subprocess.

        This endpoint attempts to gracefully terminate the global experiment
        process tracked by ``process``. It first tries a polite termination
        (SIGTERM), waits briefly, and then forcefully kills the process group
        (SIGKILL) if necessary. If ``psutil`` is available, it also attempts
        to terminate any descendant processes. Log files associated with the
        process are closed if present. All access is synchronized with
        ``process_lock``.

        Returns
        -------
        tuple[flask.Response, int]
            A JSON response indicating the result of the stop operation:
            - ``stopped``: process terminated normally.
            - ``forcefully stopped``: process required forced termination.
            - ``not running``: no experiment process was active.
            - ``unexpected error``: an exception occurred during stopping, with
            details included in the response.
        """
        global process
        with process_lock:
            status = stop_process(
                process, close_call=close_flush_files, debug_tree_bool=True
            )
            process = None  # clear global reference
            return jsonify({"status": status}), 200

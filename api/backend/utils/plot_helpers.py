# /api/backend/utils/plot_helpers.py

from flask_socketio import SocketIO


def emit_plot_bundle(
    plot_configs: list[dict], data_sources: dict, socketio: SocketIO
) -> None:
    """
    Auxiliar function for the update_plots_bundle endpoint.
    Processes a list of plot configurations and emits the corresponding data.

    This helper function iterates through each plot configuration, retrieves the
    associated data from the data_sources dictionary, and emits it as a SocketIO
    event. The event name is dynamically constructed from the plot's type and ID.

    Parameters
    ----------
    plot_configs: list[dict]
        A list of plot configuration dictionaries. Each dictionary
        is expected to have 'id', 'type', and 'src' keys.
    data_sources: dict
        A dictionary mapping source keys to data packets. Each packet
        should contain 'data' and optional 'meta' information.
    socketio: SocketIO
         The SocketIO server instance for emitting events.
    """
    for config in plot_configs:
        plot_id = config["id"]
        plot_type = config["type"]
        src_key = config["src"]

        if src_key not in data_sources:
            continue

        packet = data_sources[src_key]
        final_message = {
            "data": packet.get("data", {}),
            "meta": packet.get("meta") or {},
        }
        final_message["meta"]["plot_id"] = plot_id
        final_message["meta"]["plot_type"] = plot_type
        emit_id = f"{plot_type}-{plot_id}"

        socketio.emit(emit_id, final_message)

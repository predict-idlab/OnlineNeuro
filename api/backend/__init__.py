from pathlib import Path

from flask import Flask
from flask_socketio import SocketIO


def create_app():
    """
    Create and configure the Flask app with Socket.IO support.

    Sets up the Flask app with template and static folders, initializes
    Socket.IO for real-time communication, and registers all routes.

    Returns
    -------
    tuple[Flask, SocketIO]
        The Flask app and its associated Socket.IO instance.
    """
    base_dir = Path(__file__).resolve().parent.parent  # parent of api/ folder

    app = Flask(
        __name__,
        template_folder=str(base_dir / "frontend" / "templates"),
        static_folder=str(base_dir / "frontend" / "static"),
    )
    socketio = SocketIO(app, async_mode="eventlet")
    from .routes import register_routes

    register_routes(app, socketio)

    return app, socketio

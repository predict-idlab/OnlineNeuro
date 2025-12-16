# api/app.py
"""
Flask + Socket.IO server for OnlineNeuro experiments.

Defines all API endpoints for:
- Experiment management (start, stop, status)
- Parameter retrieval (problem, model, optimizer)
- Plot configuration and updates
- Data emission to client-side plots

Handles experiment subprocesses, background monitoring, and
real-time data streaming.

Run with:
    python3 "api/app.py" "--port" "$FLASK_PORT"
Or with shell:
    sh run.sh
"""
import argparse
import time
from pathlib import Path

from backend import create_app

start_time = time.time()

DEFAULT_PORT = 10000
LOG_DIR = Path("./tmp/experiment_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    """
    Main entry point for the Flask application.

    Initializes the Flask app and Socket.IO, parses command-line arguments
    for the port, and starts the Socket.IO server.
    """

    app, socketio = create_app()

    parser = argparse.ArgumentParser(description="Process arguments")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Flask port.")
    args = parser.parse_args()

    print("Starting Flask app...")
    print(f"http://localhost:{args.port}")
    print(app)
    print(socketio)
    socketio.run(app, port=args.port, debug=True)

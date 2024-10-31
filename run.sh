#!/bin/bash
# Get the current working directory
CURRENT_DIR=$(pwd)
DEFAULT_PORT=9000
FLASK_PORT="${1:-$DEFAULT_PORT}"

# Set the PYTHONPATH to include the current directory and necessary subdirectories
export PYTHONPATH="$WORKING_DIR:$WORKING_DIR/common:$WORKING_DIR/backend:$WORKING_DIR/frontend:$WORKING_DIR/online_neuro"

# Py version check
python3 -c "import sys; print('Python %s on %s' % (sys.version, sys.platform))"
python3 -c "import os; print(os.getenv('PYTHONPATH'))"

# Run the app
python3 "frontend/app.py" "--port" "$FLASK_PORT"

@echo off
REM Get the current working directory
set "CURRENT_DIR=%cd%"
set "DEFAULT_PORT=9000"
set "FLASK_PORT=%~1"
if "%FLASK_PORT%"=="" set "FLASK_PORT=%DEFAULT_PORT%"

REM Set the PYTHONPATH to include the current directory and necessary subdirectories
set "PYTHONPATH=%CURRENT_DIR%;%CURRENT_DIR%\common;%CURRENT_DIR%\backend;%CURRENT_DIR%\frontend;%CURRENT_DIR%\online_neuro"
set PYTHONPATH

REM Python version check
python -c "import sys; print(f'Python {sys.version} on {sys.platform}')"
python -c "import os; print(os.getenv('PYTHONPATH'))"

REM Run the app
python "frontend\app.py" --port %FLASK_PORT%

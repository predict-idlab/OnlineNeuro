# /online_neuro/utils/matlab_utils.py
"""
Utilities for interacting with the MATLAB environment and launching MATLAB processes.

This module provides the necessary functionality to start the MATLAB engine
as a subprocess and execute specific MATLAB functions, allowing the Python
optimization controller to launch and communicate with MATLAB-based simulators.
It gracefully handles cases where the required MATLAB Engine API for Python
is not installed.
"""

import json
from pathlib import Path

try:
    import matlab.engine

    MATLAB_AVAILABLE = True
except ImportError:
    MATLAB_AVAILABLE = False

if MATLAB_AVAILABLE:
    import matlab

    def run_matlab(
        matlab_script_path: str | Path, matlab_function_name: str = "main", **kwargs
    ):
        """
        Launches an external MATLAB script using the MATLAB Engine API for Python.

        This function initializes a new MATLAB engine instance, changes the current
        directory to the specified script path, adds the project root to the MATLAB
        path, and executes the specified MATLAB function, passing configurations
        as a JSON string argument.

        Parameters
        ----------
        matlab_script_path : str | Path
            The path to the directory containing the primary MATLAB script
            (e.g., 'simulators/matlab'). Can be absolute or relative to the
            project root.
        matlab_function_name : str, optional
            The name of the MATLAB function to execute (typically the entry point).
            Defaults to "main".
        **kwargs : Any
            Keyword arguments containing configurations (like connection details,
            problem setup). These are serialized into a single JSON string argument
            for the MATLAB function.

        Returns
        -------
        None

        Raises
        ------
        ImportError
            If `matlab.engine` is not installed (checked even if `MATLAB_AVAILABLE`
            was true earlier, as a safeguard).
        FileNotFoundError
            If the resolved MATLAB script directory does not exist.
        AssertionError
            If the MATLAB engine fails to start.
        """

        if not MATLAB_AVAILABLE:
            raise ImportError(
                "matlab.engine is not installed. Install it to be able to use this function"
            )
        parent_directory = Path(__file__).resolve().parents[2]

        if Path(matlab_script_path).is_absolute():
            matlab_script_full_path = Path(matlab_script_path)
        else:
            # If it's a relative path, join with current_directory
            matlab_script_full_path = parent_directory / matlab_script_path

        print("full script path:", matlab_script_full_path)
        if not matlab_script_full_path.exists():
            raise FileNotFoundError(
                f"The MATLAB folder does not exist: {matlab_script_full_path}"
            )

        matlab_script_full_path = str(matlab_script_full_path)
        # Start MATLAB engine
        print("Calling Matlab from Python engine...")

        eng = matlab.engine.start_matlab()
        assert eng is not None, "MATLAB engine failed to start"
        eng.cd(matlab_script_full_path)
        eng.addpath(str(parent_directory), nargout=0)

        # Serialize Python arguments into a JSON string for MATLAB
        matlab_args = json.dumps(kwargs)

        try:
            # Call the main MATLAB function
            eng.feval(matlab_function_name, matlab_args, nargout=0)
            eng.quit()

        except Exception as e:
            print(f"Error starting MATLAB: {e}")
            raise

else:

    def run_matlab(
        matlab_script_path: str | Path, matlab_function_name: str = "main", **kwargs
    ):
        """
        Placeholder function for launching a MATLAB process when the MATLAB Engine API
        is not available.

        Raises
        ------
        ImportError
            Always raises an ImportError, instructing the user to install the
            required MATLAB engine library.
        """
        raise ImportError(
            "matlab.engine is not installed. Install it to be able to use this function"
        )

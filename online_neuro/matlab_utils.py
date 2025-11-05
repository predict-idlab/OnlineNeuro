# /online_neuro/matlab_utils.py
import json
from pathlib import Path

try:
    import matlab.engine

    MATLAB_AVAILABLE = True
except ImportError:
    MATLAB_AVAILABLE = False

if MATLAB_AVAILABLE:
    import matlab

    def run_matlab(matlab_script_path, matlab_function_name="main", **kwargs):
        """
        Launch a matlab process
        @param matlab_script_path:
        @param matlab_function_name:
        @return:

        """
        if not MATLAB_AVAILABLE:
            raise ImportError(
                "matlab.engine is not installed. Install it to be able to use this function"
            )

        current_directory = Path(__file__).resolve().parent
        parent_directory = current_directory.parent

        if Path(matlab_script_path).is_absolute():
            matlab_script_full_path = Path(matlab_script_path)
        else:
            # If it's a relative path, join with current_directory
            # Here we assume that problems will be stored at ./simulators/matlab/problems
            matlab_script_full_path = parent_directory / matlab_script_path

        if not matlab_script_full_path.exists():
            raise FileNotFoundError(
                f"The MATLAB folder does not exist: {matlab_script_full_path}"
            )

        matlab_script_full_path = str(matlab_script_full_path)
        # Start MATLAB engine
        print("Calling Matlab from Python engine")

        eng = matlab.engine.start_matlab()
        eng.cd(matlab_script_full_path)
        eng.addpath(str(parent_directory), nargout=0)
        # Call MATLAB function main
        matlab_args = json.dumps(kwargs)

        try:
            eng.feval(matlab_function_name, matlab_args, nargout=0)
            eng.quit()

        except Exception as e:
            print(f"Error starting MATLAB: {e}")

else:

    def run_matlab(matlab_script_path, matlab_function_name="main", **kwargs):
        """
        Launch a matlab process
        @param matlab_script_path:
        @param matlab_function_name:
        @return:

        """
        raise ImportError(
            "matlab.engine is not installed. Install it to be able to use this function"
        )

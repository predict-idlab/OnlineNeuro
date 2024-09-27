# frontend/components/config_forms.py
import warnings
import streamlit as st
from common.utils import load_json


def config_problem(problem: str='Axonsim') -> dict:
    """
    @param problem: String, to define the type of problem that will be solved (examples or simulators)
    @return: config: dict containing the selected parameters
    """
    # Dynamically create input widgets based on JSON keys
    config = {}
    # Load the JSON file
    if problem.startswith("Axonsim"):
        json_file = "config_experiments/axonsim_template.json"
        json_data = load_json(json_file)

        config["num_electrodes"] = st.sidebar.number_input("Select the number of pulses",
                                                           min_value=1,
                                                           max_value=3,
                                                           value=1,
                                                           step=1)

        col1, col2 = st.sidebar.columns(2)
        # Removing non-dict keys (these are just there to display the different type of "inputs"
        filt_keys = [k for k in json_data.keys() if isinstance(json_data[k], dict)]

        for key in filt_keys:
            # TODO specify step size
            # TODO handle e_pos
            if 'optimizable' in json_data[key] and json_data[key]['optimizable']:
                for i in range(config["num_electrodes"]):
                    if i < len(json_data[key]['value']):
                        config[f"{key}_{i + 1}_min"] = col1.number_input(f"{key} {i} min",
                                                                         value=json_data[key]['value'][i],
                                                                         min_value=json_data[key]['min_value'][i],
                                                                         max_value=json_data[key]['max_value'][i], )
                        config[f"{key}_{i + 1}_max"] = col2.number_input(f"{key} {i} max",
                                                                         value=json_data[key]['value'][i],
                                                                         min_value=json_data[key]['min_value'][i],
                                                                         max_value=json_data[key]['max_value'][i])

                    else:
                        msg = f"Config files does not contain enough cols, defaulting values for {key}"
                        config[f"{key}_{i + 1}_min"] = col1.number_input(f"{key} {i} min",
                                                                         value=json_data[key]['value'][0],
                                                                         min_value=json_data[key]['min_value'][0],
                                                                         max_value=json_data[key]['max_value'][0], )
                        config[f"{key}_{i + 1}_max"] = col2.number_input(f"{key} {i} max",
                                                                         value=json_data[key]['value'][0],
                                                                         min_value=json_data[key]['min_value'][0],
                                                                         max_value=json_data[key]['max_value'][0])
                        warnings.warn(msg)
    elif problem.endswith("Regression"):
        pass
    elif problem.endswith("Classification"):
        json_file = "config_experiments/circle_template.json"
        json_data = load_json(json_file)

    elif problem.endswith("MOO"):
        pass
    else:
        #Placeholder, just for debugging purposes
        json_file = "config_experiments/dummy.json"
        json_data = load_json(json_file)

        for key, value in json_data.items():
            if isinstance(value, list):  # If value is a list, create a dropdown
                config[key] = st.sidebar.selectbox(f"Select {key}", value)
            elif isinstance(value, bool):  # If value is boolean, create a checkbox
                config[key] = st.sidebar.checkbox(f"Toggle {key}", value)
            elif isinstance(value, str):  # If value is a string, create a text input
                config[key] = st.sidebar.text_input(f"Enter {key}", value)
            elif isinstance(value, int):  # If value is an integer, create a number input
                config[key] = st.sidebar.number_input(f"Set {key}", value)
            elif isinstance(value, float):  # If value is a float, create a number input
                config[key] = st.sidebar.number_input(f"Set {key}", value, format="%.2f")

    return config

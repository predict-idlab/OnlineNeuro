# frontend/components/config_forms.py
import warnings
from common.utils import load_json
from pathlib import Path
# May need some rework or such, as it may lead to a very long list
# # or require to constantly update this dictionary

template_map = {
    "axonsim_nerve_block": Path("config") / "experiments" / "axonsim_template.json",
    "axonsim_regression": Path("config") / "experiments" / "axonsim_template.json",
    "rose_regression": Path("config") / "experiments" / "rose_template.json",
    "circle_classification": Path("config") / "experiments" / "circle_template.json",
    "vlmop2": Path("config") / "experiments" / "vlmop2_template.json",
    "cajal_ap_block": Path("config") / "experiments" / "cajal_template.json"
}

function_map = {
    'pulse_ramp': Path("config") / "custom_pulses" / "pulse_ramp.json",
    'monophasic': Path("config") / "custom_pulses" / "monophasic.json",
    'default': Path("config") / "custom_pulses" / "axonsim_default.json"
}

matlab_experiments = {"Axonsim (nerve block)": "axonsim_nerve_block",
                       "Axonsim (regression)": "axonsim_regression",
                       "Toy Regression": "rose_regression",
                       "Toy Classification": "circle_classification",
                       "Toy VLMOP2": "vlmop2"
                      }

experiments_types = {
    "axonsim_nerve_block": "classification",
    "cajal_ap_block":"classification",
    "axonsim_regression": "regression",
    "rose_regression": "regression",
    "circle_classification": "classification",
    "vlmop2": "moo"
}

python_experiments = {"Toy classification (Python)": "circle_classification",
                      #"Toy regression (Python)": "rose_regression",
                      "Toy MOO (Python)": "vlmop2",
                      "Cajal nerve block": "cajal_ap_block"}

#Dictionary to validate model problem and model type.
# VALID_PROBLEMS = {
#     'axonsim_nerve_block': ['classification', 'regression', 'moo'],
#     'axonsim_regression': ['regression', 'moo'],
#     'cajal_nerve_block': ['classification', 'regression', 'moo'],
#     'todo': ['placeholder']
# }


def verify_optim_fixed(config):
    for k, v in config.items():
        if isinstance(v, dict):
            if ('optimizable' in v) and ('user_fixed' in v):
                assert (v['optimizable'] != v['user_fixed']) or (v['optimizable'] is False and v['user_fixed'] is False)


def config_function(function: str) -> dict:
    """
    @param function: String, name of the function we require parameters for
    @return:
    """
    function = function.lower()
    if function in function_map:
        config = load_json(function_map[function])
    else:
        config = load_json(function_map['default'])

    verify_optim_fixed(config)
    return config


def config_problem(problem: str = 'axonsim') -> dict:
    """
    @param problem: String, to define the type of problem that will be solved (examples or simulators)
    @return: config: dict containing the selected parameters
    """
    # TODO. do this in a smart way.
    # Dynamically create input widgets based on JSON keys
    # Load the JSON file
    problem = problem.lower()
    try:
        config = load_json(template_map[problem])
        # TODO as they exclude each other this can be simplified!
        verify_optim_fixed(config)

        config['experiment'] = dict()
        config['experiment']['value'] = problem

    except NotImplementedError:
        msg = f"No configuration file for experiment: ({problem})"
        raise msg

    return config


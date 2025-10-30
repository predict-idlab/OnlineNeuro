# api/frontend/components/config_forms.py
from pathlib import Path

from common import utils as nn_utils

# May need some rework or such, as it may lead to a very long list
# or require to constantly update this dictionary

template_map = {
    "axonsim_nerve_block": Path("config") / "experiments" / "axonsim_template.json",
    "axonsim_regression": Path("config") / "experiments" / "axonsim_template.json",
    "rose_regression": Path("config") / "experiments" / "rose_template.json",
    "circle_classification": Path("config") / "experiments" / "circle_template.json",
    "vlmop2": Path("config") / "experiments" / "vlmop2_template.json",
    "cajal_ap_block": Path("config") / "experiments" / "cajal_template.json",
}

model_map = {
    "mc_dropout_nn": Path("config") / "models" / "mc_nn.json",
    "ensemble_nn": Path("config") / "models" / "ensemble_nn.json",
    "gp": Path("config") / "models" / "gp.json",
}

add_model_map = {
    "nn": Path("config") / "models" / "nn.json",
    "common": Path("config") / "models" / "common.json",
    "common_nn": Path("config") / "models" / "common_nn.json",
}

function_map = {
    "pulse_ramp": Path("config") / "custom_pulses" / "pulse_ramp.json",
    "monophasic": Path("config") / "custom_pulses" / "monophasic.json",
    "default": Path("config") / "custom_pulses" / "axonsim_default.json",
}

optimizer_config_path = Path("config") / "optimization" / "common.json"

matlab_experiments = {
    "Axonsim (nerve block)": "axonsim_nerve_block",
    "Axonsim (regression)": "axonsim_regression",
    "Toy Regression": "rose_regression",
    "Toy Classification": "circle_classification",
    "Toy VLMOP2": "vlmop2",
}

python_experiments = {
    "Toy classification (Python)": "circle_classification",
    # "Toy regression (Python)": "rose_regression",
    "Toy MOO (Python)": "vlmop2",
    "Cajal nerve block": "cajal_ap_block",
}

experiments_types = {
    "axonsim_nerve_block": "classification",
    "axonsim_regression": "regression",
    "cajal_ap_block": "classification",
    "rose_regression": "regression",
    "circle_classification": "classification",
    "vlmop2": "moo",
}

# Dictionary to validate model problem and model type.
# VALID_PROBLEMS = {
#     'axonsim_nerve_block': ['classification', 'regression', 'moo'],
#     'axonsim_regression': ['regression', 'moo'],
#     'cajal_nerve_block': ['classification', 'regression', 'moo'],
#     'todo': ['placeholder']
# }


def verify_optim_fixed(config: dict) -> None:
    for k, v in config.items():
        if isinstance(v, dict):
            if ("optimizable" in v) and ("user_fixed" in v):
                assert (v["optimizable"] != v["user_fixed"]) or (
                    v["optimizable"] is False and v["user_fixed"] is False
                )


def config_function(function: str) -> dict:
    """
    @param function: String, name of the function we require parameters for
    @return:
    """
    function = function.lower()
    if function in function_map:
        config = nn_utils.load_json(function_map[function])
    else:
        config = nn_utils.load_json(function_map["default"])

    verify_optim_fixed(config)
    return config


def config_problem(problem: str = "axonsim") -> dict:
    """
    @param problem: String, to define the type of problem that will be solved (examples or simulators)
    @return: config: dict containing the selected parameters
    """
    # TODO. do this in a smart way.
    # Dynamically create input widgets based on JSON keys
    # Load the JSON file
    problem = problem.lower()
    try:
        config = nn_utils.load_json(template_map[problem])
        # TODO as they exclude each other this can be simplified!
        verify_optim_fixed(config)

        config["experiment"] = dict()
        config["experiment"]["value"] = problem

    except NotImplementedError:
        msg = f"No configuration file for experiment: ({problem})"
        raise msg

    return config


def config_model(model: str) -> dict:
    """
    @param model: String, to define the type of model that will be used.
    @return: config: dict for the model(s)
    """
    # TODO. do this in a smart way.
    # Dynamically create input widgets based on JSON keys
    # Load the JSON file
    model = model.lower()
    try:
        config = nn_utils.load_json(model_map[model])
    except NotImplementedError:
        msg = f"No configuration file for model: {model} ({model_map[model]})"
        raise NotImplementedError(msg)

    try:
        common = nn_utils.load_json(add_model_map["common"])
    except ValueError:
        msg = f" Error while loading common model file. ({add_model_map['common']})"
        raise ValueError(msg)

    if model in ["mc_dropout_nn", "ensemble_nn"]:
        nn_common = nn_utils.load_json(add_model_map["common_nn"])

        for k, v in nn_common.items():
            if k not in config:
                config[k] = v
            else:
                msg = f"Keys in NN configuration present in {model} json configuration file."
                raise Exception(msg)

    for k, v in common.items():
        if k not in config:
            config[k] = v
        else:
            if model in ["mc_dropout_nn", "ensemble_nn"]:
                if k in nn_common:
                    msg = f"Key {k} in common_nn json configuration file"
                    raise Exception(msg)

            msg = f"Key {k} in common present in {model} configuration file"
            raise Exception(msg)
    return config


def config_optimizer() -> dict:
    try:
        config = nn_utils.load_json(optimizer_config_path)
    except Exception as e:
        msg = f"Failed at trying to load the optimizar configuration file: {e}"
        raise NotImplementedError(msg)

    return config

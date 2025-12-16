# /simulators/python/problems/cajal_problems.py
from typing import Any

import numpy as np
from cajal.nrn import Backend as CajalBackend
from cajal.nrn import SimulationEnvironment
from cajal.nrn.cells import MRG
from cajal.nrn.monitors import APMonitor, StateMonitor
from cajal.nrn.sources import IsotropicPoint, Point
from cajal.nrn.stimuli import MonophasicPulse, Stimulus
from cajal.units import cm, mm, ms, mV, ohm

from common.decorators import ensure_lists
from simulators.python.processing.cajal_processing import ap_block
from simulators.python.pulses.pulse_definitions import PulseRamp


def generate_circular_arc(
    radius: float,
    num_nodes: int,
    total_angle: float,
    plane: str = "xy",
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    start_angle: float = 0.0,
    degrees: bool = False,
) -> list[tuple[float, float, float]]:
    """
    Generate positions for a curved axon following a circular arc.

    Parameters
    ----------
    radius : float
        Arc radius.
    num_nodes : int
        Number of discrete nodes along the arc.
    total_angle : float
        Arc angle (radians by default).
    plane : {'xy', 'xz', 'yz'}, optional
        Plane in which to generate the arc (default is 'xy').
    center : tuple of 3 floats, optional
        Center of the circle (cx, cy, cz). Default is (0, 0, 0).
    start_angle : float, optional
        Starting angle of the arc (default is 0).
    degrees : bool, optional
        Interpret angles as degrees if True (default is False).

    Returns
    -------
    positions : list of tuple of float
        List of 3D positions (x, y, z) along the arc.
    """

    if degrees:
        total_angle = np.deg2rad(total_angle)
        start_angle = np.deg2rad(start_angle)

    angles = np.linspace(start_angle, start_angle + total_angle, num_nodes)
    cx, cy, cz = center

    positions = []
    for theta in angles:
        if plane == "xy":
            x = cx + radius * np.cos(theta)
            y = cy + radius * np.sin(theta)
            z = cz
        elif plane == "xz":
            x = cx + radius * np.cos(theta)
            y = cy
            z = cz + radius * np.sin(theta)
        elif plane == "yz":
            x = cx
            y = cy + radius * np.cos(theta)
            z = cz + radius * np.sin(theta)
        else:
            raise NotImplementedError(
                f"plane {plane} is not valid. Select between 'xy', 'xz' and 'yz'"
            )

        positions.append((x, y, z))

    return positions


def create_mrg(verbose: bool = False, **kwargs: Any):
    """
    Create an MRG axon model with specified parameters.

    Parameters
    ----------
    verbose : bool, optional
        If True, prints detailed information about the axon, including number
        of nodes, total length, and 3D coordinates of first and last nodes.
        Default is False.
    **kwargs
        Keyword arguments to pass to the MRG constructor.

    Returns
    -------
    MRG
        An instance of the MRG axon model.

    Notes
    -----
    When `verbose` is True:
    - Prints the computed number of nodes.
    - Prints the total axon length in millimeters.
    - Prints the 3D coordinates (x, y, z) in millimeters for the first and last nodes.

    Examples
    --------
    >>> mrg_model = create_mrg(verbose=True, diameter=10, length=1000)
    Computed number of Nodes: 200
    Computed Axon Length: 10.00 mm
    Node position of the first node in the axon: x=0.00 mm, y=0.00 mm, z=0.00 mm
    Node position of the last node in the axon: x=0.50 mm, y=10.00 mm, z=0.50 mm
    """

    mrg = MRG(**kwargs)

    # If verbose mode is enabled, output detailed information about the axon
    if verbose:
        # Print the total number of nodes in the axon
        print(f"Computed number of Nodes: {len(mrg.node)}")

        # Calculate and print the total axon length based on the first and last node positions
        first_node_y = mrg.node[0].y3d(0)
        last_node_y = mrg.node[-1].y3d(0)
        total_length = (last_node_y - first_node_y) / 1000
        print(f"Computed Axon Length: {total_length:.2f} mm")

        # Print the 3D coordinates of the first and last nodes
        for node, node_name in zip([mrg.node[0], mrg.node[-1]], ["first", "last"]):
            x = node.x3d(0) / 1000  # Convert from um to mm
            y = node.y3d(0) / 1000
            z = node.z3d(0) / 1000
            print(
                f"Node position of the {node_name} node in the axon: x={x:.2f} mm, y={y:.2f} mm, z={z:.2f} mm"
            )

    return mrg


def create_monitors(mrg: MRG, ap_node_0: int = 0, ap_node_1: int = 1) -> dict[str, Any]:
    """
    Create action potential (AP) and state monitors for a given MRG axon model.

    Parameters
    ----------
    mrg : MRG
        The MRG axon model instance.
    ap_node_0 : int, optional
        Index of the first node for the APMonitor (default is 0).
    ap_node_1 : int, optional
        Index of the second node for the APMonitor (default is 1).

    Returns
    -------
    monitors : dict
        Dictionary containing:
        - 'node_0': APMonitor for the first node.
        - 'node_1': APMonitor for the second node.
        - 'state': StateMonitor recording the membrane potential 'v' for all nodes.

    Notes
    -----
    - APMonitor triggers when the voltage crosses a threshold (-20 mV by default).
    - StateMonitor records the membrane potential across all nodes.
    - AP monitors may be redundant if state recording and postprocessing are sufficient.

    Examples
    --------
    >>> monitors = create_monitors(mrg_model, ap_node_0=0, ap_node_1=1)
    >>> monitors['state']  # StateMonitor object for the MRG nodes
    >>> monitors['node_0']  # APMonitor for the first node
    """

    ap_monitor_0 = APMonitor(mrg.node[ap_node_0], threshold=-20 * mV)
    ap_monitor_1 = APMonitor(mrg.node[ap_node_1], threshold=-20 * mV)
    # or State monitors
    v_rec = StateMonitor(mrg.node, "v")

    monitors = {"node_0": ap_monitor_0, "node_1": ap_monitor_1, "state": v_rec}

    return monitors


def create_pulse(type: str, *args: Any, **kwargs: Any) -> Stimulus:
    """
    Create a stimulus pulse of a specified type.

    Parameters
    ----------
    stim_type : str
        Type of stimulus to create. Supported types:
        - 'monophasic' or 'single_pulse'
        - 'pulse_ramp' or 'pulseramp'
    *args
        Positional arguments passed to the pulse constructor.
    **kwargs
        Keyword arguments passed to the pulse constructor.
        For monophasic pulses, commonly used arguments include:
        - amp or amplitude : float
            Pulse amplitude
        - pw or pulse_width : float
            Pulse width
        - delay : float, optional
            Delay before pulse starts

    Returns
    -------
    stim : Stimulus, currently MonophasicPulse or PulseRamp
        Instance of the requested stimulus pulse.

    Raises
    ------
    NotImplementedError
        If `stim_type` is not recognized.

    Examples
    --------
    >>> pulse = create_pulse('monophasic', amp=1.0, pw=0.1)
    >>> pulse = create_pulse('pulse_ramp', start=0, stop=1, duration=0.5)
    """

    if type in ["monophasic", "single_pulse"]:
        amp = kwargs.get("amp", kwargs.get("amplitude", None))
        pw = kwargs.get("pw", kwargs.get("pulse_width", None))
        delay = kwargs.get("delay", None)
        stim = MonophasicPulse(amp=amp, pw=pw, delay=delay)

    elif type in ["pulse_ramp", "pulseramp"]:
        stim = PulseRamp(**kwargs)

    else:
        msg = f"Stimulus type {type} not implemented"
        raise NotImplementedError(msg)

    return stim


def create_stimulus(
    pos: list[list[float]],
    stimulus_params: list[dict[str, Any]] | dict[str, Any],
    rhoe: float = 500,
    source_type: str = "isotropic",
    **kwargs: Any,
) -> tuple[list[Point], list[Stimulus]]:
    """
    Create electrodes and stimulus pulses for multiple stimulation sites.

    Parameters
    ----------
    pos : list of list of float
        List of electrode 3D positions [[x, y, z], ...] in mm.
    stimulus_params : dict or list of dict
        Dictionary or list of dictionaries containing parameters for each pulse.
        Each dict may contain 'fun_type' key for pulse type and other keyword
        arguments for `create_pulse`.
    rhoe : float, optional
        Extracellular resistivity in ohm·cm (default is 500).
    source_type : str, optional
        Type of point source. Only 'isotropic' is implemented (default).
    **kwargs
        Additional keyword arguments (not used directly).

    Returns
    -------
    electrodes : list
        List of electrode objects (point source combined with stimulus pulse).
    stimuli : list
        List of stimulus pulse objects created for each electrode.

    Raises
    ------
    ValueError
        If the number of stimulus parameter dictionaries does not match the
        number of electrode positions.
    NotImplementedError
        If `source_type` is not implemented.

    Examples
    --------
    >>> positions = [[0, 0, 0], [1, 0, 0]]
    >>> params = [{'fun_type': 'monophasic', 'amp': 1, 'pw': 0.1}, {'fun_type': 'pulse_ramp', 'start':0, 'stop':1}]
    >>> electrodes, stimuli = create_stimulus(positions, params)
    """

    num_electrodes = len(pos)

    if not isinstance(stimulus_params, list):
        stimulus_params = [stimulus_params]

    if len(stimulus_params) != num_electrodes:
        raise ValueError(
            f"stimulus_params format is incorrect. Expected {num_electrodes} "
            f"dictionaries, but got {len(stimulus_params)}."
        )

    params_list = stimulus_params

    electrodes = []
    stimuli = []

    for electrode_pos, electrode_params in zip(pos, params_list):
        # --- Create the point source for this electrode ---
        if source_type.lower() == "isotropic":
            point_source = IsotropicPoint(
                x=electrode_pos[0] * mm,
                y=electrode_pos[1] * mm,
                z=electrode_pos[2] * mm,
                rhoe=rhoe * ohm * cm,
            )
        else:
            raise NotImplementedError(
                f"Point Source type: '{source_type}' not implemented"
            )

        # --- Create the stimulus pulse for this electrode ---
        # Make a copy to avoid modifying the original input dict
        params_for_pulse = electrode_params.copy()

        # Pop 'fun_type' to get the pulse type, and the rest of the dict
        # is passed as keyword arguments to create_pulse.
        pulse_type = params_for_pulse.pop(
            "fun_type", "monophasic"
        )  # Default if not provided
        stim = create_pulse(type=pulse_type, **params_for_pulse)
        stimuli.append(stim)

        # --- Combine the source and stimulus to create the electrode ---
        electrode = point_source << stim
        electrodes.append(electrode)

    return electrodes, stimuli


@ensure_lists
def create_simulation(axons: Any, stimulus: Any, monitors: Any):
    """
    Create a simulation environment for one or more axons with specified stimuli and monitors.

    Parameters
    ----------
    axons : object or list of objects
        Single axon model or list of axon models to simulate.
    stimulus : object or list of objects
        Single stimulus or list of stimuli to apply to the axons.
    monitors : object or list of objects
        Single monitor or list of monitors for recording axon activity.

    Returns
    -------
    SimulationEnvironment
        An instance of the simulation environment containing the specified axons,
        stimuli, and monitors.

    Notes
    -----
    - The `@ensure_lists` decorator ensures that `axons`, `stimulus`, and `monitors`
      are always treated as lists internally, even if a single object is provided.

    Examples
    --------
    >>> sim_env = create_simulation(axon_model, stimulus_obj, monitors_dict)
    >>> sim_env.run()  # Run the simulation
    """
    env = SimulationEnvironment(axons=axons, extra_stim=stimulus, monitors=monitors)
    return env


def cajal_fun(
    sim_dur: float = 5.0,
    sim_early_stopping: bool = False,
    fun_type: list = [],
    axon_diameter: float = 5.7,
    axon_length: float = 35.0,
    post_processing: str = "ap_block",
    verbose: bool = True,
    **kwargs,
) -> dict[str, Any]:
    """
    Run a single (axon) simulation with specified stimulus and return processed results.

    Parameters
    ----------
    sim_dur : float, optional
        Simulation duration in milliseconds (default 5.0 ms).
    sim_early_stopping : bool, optional
        If True, stop the simulation early when criteria are met (default False).
    fun_type : list of str, optional
        List specifying pulse types for each electrode (default empty list).
    axon_diameter : float, optional
        Axon diameter in micrometers (default 5.7).
    axon_length : float, optional
        Axon length in millimeters (default 35.0).
    post_processing : str, optional
        Method for post-processing (currently only "ap_block" supported).
    verbose : bool, optional
        If True, prints parameters and diagnostic information (default True).
    **kwargs
        Additional keyword arguments:
        - Any parameter starting with "axon_" is treated as axon model parameter.
        - Any parameter starting with "e_" is treated as electrode parameter.
        - Any parameter starting with "pulse_parameters_" is treated as pulse parameter.
        - "time_resolution" can override CajalBackend.dt.

    Returns
    -------
    data_package : dict
        Dictionary containing:
        - 'observations': processed result of post-processing (e.g., 0 or 1 for AP block)
        - 'full_observations': dictionary with full voltage traces, time, and pulse signals

    Raises
    ------
    NotImplementedError
        If `post_processing` method is not recognized.

    Notes
    -----
    - This function builds the MRG axon model, electrodes, stimuli,
      and monitors based on the provided parameters.
    - Post-processing currently only supports AP block detection.
    """

    # Override time resolution if provided
    if "time_resolution" in kwargs:
        CajalBackend.dt = kwargs["time_resolution"] * ms

    # Collect all parameters
    all_params = {key: value for key, value in locals().items() if key != "kwargs"}
    all_params = {**all_params, **kwargs}

    # Simulation parameters
    sim_params = {
        "dur": sim_dur,
        "early_stopping": sim_early_stopping,
    }

    # Axon parameters
    axon_params = {
        k.replace("axon_", ""): v
        for k, v in all_params.items()
        if k.startswith("axon_")
    }
    axon_params.update({"axon_diameter": axon_diameter, "axon_length": axon_length})

    axon_params = {"_".join(k.split("_")[1:]): v for k, v in axon_params.items()}

    # Electrode parameters
    electrode_params = {
        k.replace("e_", ""): v for k, v in all_params.items() if k.startswith("e_")
    }

    stimulus_params: dict[int, dict[str, Any]] = dict()
    flat_params = {
        k: v
        for k, v in all_params.items()
        if k.startswith("pulse_parameters_") or k.startswith("s_")
    }
    for key, value in flat_params.items():
        if key.startswith("pulse_parameters_"):
            parts = key.split("_")  # ['pulse', 'parameters', '1', 's', 'pw']
            idx = int(parts[2])  # electrode index
            name = "_".join(parts[4:])  # 's_pw', 's_amp', etc.
            stimulus_params.setdefault(idx, {})[name] = value

    stimulus_params_list = [stimulus_params[i] for i in sorted(stimulus_params.keys())]
    for e, d in enumerate(stimulus_params_list):
        d["fun_type"] = fun_type[e]

    axon_model = create_mrg(**axon_params)
    monitors = create_monitors(axon_model)
    electrode_params["pos"] = np.array(electrode_params["pos"]).T.tolist()

    if verbose:
        print("Axon params:", axon_params)
        print("Stimulus params:", stimulus_params_list)
        print("Electrode params:", electrode_params)

    stimulus, pulses = create_stimulus(
        pos=electrode_params["pos"],
        rhoe=electrode_params["rhoe"],
        stimulus_params=stimulus_params_list,
        monitors=monitors,
    )

    env = create_simulation(
        axons=axon_model, stimulus=stimulus, monitors=list(monitors.values())
    )

    env.run(sim_params["dur"] * ms, early_stopping=sim_params["early_stopping"])

    if post_processing == "ap_block":
        y = ap_block(
            ap_monitor_stimulus=monitors["node_0"],
            ap_monitor_block=monitors["node_1"],
            start_time=2.50 * ms,
            end_time=3.10 * ms,
            verbose=verbose,
        )
    else:
        msg = f"Post processing method {post_processing} not implemented"
        raise NotImplementedError(msg)

    data_package = {
        "observations": y,
        "full_observations": {
            "time": monitors["state"].t.tolist(),
            "voltage": monitors["state"].v.tolist(),
            "stim_pulse": pulses[0](monitors["state"].t).tolist(),
            "block_pulse": pulses[1](monitors["state"].t).tolist(),
        },
    }

    return data_package

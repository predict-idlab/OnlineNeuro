# /simulators/python/problems/cajal_problems.py
try:
    import cajal
    CAJAL_AVAILABLE = True
except ImportError:
    raise ImportError('cajal is not installed. Install it to use this module')

if CAJAL_AVAILABLE:
    from cajal.nrn import Backend as CajalBackend
    from cajal.nrn.cells import MRG
    from cajal.units import mm, um, mV, ms, mA, ms, ohm, cm
    from cajal.nrn.stimuli import MonophasicPulse
    from cajal.nrn.sources import IsotropicPoint
    from cajal.nrn.monitors import StateMonitor, APMonitor
    from cajal.nrn import SimulationEnvironment

    import re
    import numpy as np
    from ..pulses.pulse_definitions import PulseRamp

    def generate_circular_arc(radius, num_nodes, total_angle):
        """
        TODO (finish this, specify pair of
        Generate positions for a curved axon following a circular arc.
        Returns:
        - List of (x, y, z) positions.
        """
        angles = np.linspace(0, total_angle, num_nodes)
        positions = [(radius * np.sin(theta), radius * (1 - np.cos(theta)), 0) for theta in angles]
        return positions


    def ensure_lists(func):
        def wrapper(*args, **kwargs):
            # Modify the positional arguments to ensure each is a list
            new_args = [arg if isinstance(arg, list) else [arg] for arg in args]

            # Ensure the keyword arguments
            new_kwargs = {key: (value if isinstance(value, list) else [value])
                            for key, value in kwargs.items()}

            # Call the original function with modified arguments
            return func(*new_args, **new_kwargs)

        return wrapper


    def create_mrg(verbose: bool=False,*args, **kwargs):
        # Create the axon model (MRG) with specified diameter, length, and initial voltage
        mrg = MRG(*args, **kwargs)

        # If verbose mode is enabled, output detailed information about the axon
        if verbose:
            # Print the total number of nodes in the axon
            print(f'Computed number of Nodes: {len(mrg.node)}')

            # Calculate and print the total axon length based on the first and last node positions
            first_node_y = mrg.node[0].y3d(0)
            last_node_y = mrg.node[-1].y3d(0)
            total_length = (last_node_y - first_node_y) / 1000
            print(f'Computed Axon Length: {total_length:.2f} mm')

            # Print the 3D coordinates of the first and last nodes
            for node, node_name in zip([mrg.node[0], mrg.node[-1]], ['first', 'last']):
                x = node.x3d(0) / 1000  # Convert from um to mm
                y = node.y3d(0) / 1000
                z = node.z3d(0) / 1000
                print(f'Node position of the {node_name} node in the axon: x={x:.2f} mm, y={y:.2f} mm, z={z:.2f} mm')

        return mrg

    def create_monitors(mrg, ap_node_0=0, ap_node_1=1):
        # TODO AP monitors can be added here
        ap_monitor_0 = APMonitor(mrg.node[ap_node_0], threshold=-20 * mV)
        ap_monitor_1 = APMonitor(mrg.node[ap_node_1], threshold=-20 * mV)
        # or State monitors
        v_rec = StateMonitor(mrg.node, 'v')
        # TODO in general it may be better to use StateMonitors to get a better idea of
        # what type of block is happening
        # More detailed ap_block function is needed here.

        monitors = {'node_0': ap_monitor_0,
                    'node_1': ap_monitor_1,
                    'state': v_rec}

        return monitors


    def create_pulse(type, *args, **kwargs):
        if type in ['monophasic', 'single_pulse']:
            amp = kwargs.get('amp', None)
            pw = kwargs.get('pw', None)
            delay = kwargs.get('delay', None)
            stim = MonophasicPulse(amp=amp, pw=pw, delay=delay)

        elif type in ['pulse_ramp', 'pulseramp']:
            stim = PulseRamp(**kwargs)

        else:
            raise NotImplementedError(f'Stimulus type {type} not implemented')

        return stim


    def create_stimulus(pos: list, stimulus_params, rhoe: float=500, monitors=None, **kwargs):
        point_sources = []
        for ix in range(len(pos)):
            g = pos[ix]

            if 'source_type' not in kwargs:
                # Default to Isotropic
                # TODO, allow adjusting res / rhoe
                point_source = IsotropicPoint(x=g[0] * mm,
                                                y=g[1] * mm,
                                                z=g[2] * mm,
                                                rhoe=rhoe * ohm * cm)
            elif kwargs['source_type'] == 'isotropic':
                point_source = IsotropicPoint(x=g[0] * mm,
                                                y=g[1] * mm,
                                                z=g[2] * mm,
                                                rhoe=rhoe * ohm * cm)
            else:
                NotImplementedError(f"Point Source type: {kwargs['source_type']},  not implemented")

            point_sources.append(point_source)

        num_electrodes = len(pos)

        electrodes = []
        stimuli = []
        for ix in range(num_electrodes):
            pattern = re.compile(r'_([1-9])$')
            filtered_keys = {key.split('_')[0]: value for key, value in stimulus_params.items() if pattern.search(key)}
            stim = create_pulse(type=stimulus_params['fun_type'][ix], **filtered_keys)
            stimuli.append(stim)

            electrode = point_sources[ix] << stim
            electrodes.append(electrode)

        return electrodes, stimuli


    @ensure_lists
    def create_simulation(axons, stimulus, monitors):
        env = SimulationEnvironment(
            axons=axons,
            extra_stim=stimulus,
            monitors=monitors
        )

        return env


    def ap_block(ap_monitor_stimulus, ap_monitor_block, start_time, end_time, verbose=False) -> int:
        """
        TODO Evaluate / Discuss this function as it may not be fully correct.

        @param ap_monitor_stimulus:
        @param ap_monitor_block:
        @param start_time:
        @param end_time:
        @param verbose:
        @return:
        """
        ap_count_end = ap_monitor_block.n(t=start_time)
        ap_count_start_end = ap_monitor_stimulus.n(t=end_time)
        ap_count_start_start = ap_monitor_stimulus.n(t=start_time)
        ap_count_left = ap_count_start_start - ap_count_start_end  # Never less than zero
        ap_times_end = ap_monitor_block.spikes()

        if verbose:
            print(f'Number of APs between {start_time} and {end_time} ms at start node: {ap_count_left}')
            print(f'All spike times at end node: {ap_times_end}')

        if ap_count_end > 0:
            if verbose:
                print('AP NOT blocked')
            return 1
        else:
            if ap_count_left == 0:
                if verbose:
                    print('NO AP at the left side')
                return 1
            else:
                if verbose:
                    print('AP is blocked')
                return 0


    def ap_block_2(v_monitor, threshold=20, left_node=15, right_node=-15, verbose=False) -> int:
        """
        TODO Evaluate / Discuss this function as it may not be fully correct.
        @return:
        """
        data = v_monitor.v
        max_time = np.max(data, axis=0)
        b0 = max_time[left_node] > threshold
        b1 = max_time[right_node] < threshold
        a = b0 and b1

        b2 = max_time[left_node] < threshold
        b3 = max_time[right_node] > threshold
        b = b2 and b3

        if a:
            print('Right block')
        if b:
            print('Left block (?)')
        return int(a or b)


    def cajal_fun(e_pos: list,
                    sim_dur: float = 5.0,
                    sim_early_stopping: bool = False,
                    fun_type: list = [],
                    axon_diameter: float = 5.7,
                    axon_length: float = 35,
                    post_processing: str = 'ap_block',
                    verbose: bool = False, **kwargs):

        if 'time_resolution' in kwargs:
            CajalBackend.dt = kwargs['time_resolution'] * ms

        all_params = {key: value for key, value in locals().items() if key != 'kwargs'}
        all_params = {**all_params, **kwargs}

        sim_params = {
            'dur': sim_dur,
            'early_stopping': sim_early_stopping,
        }
        axon_params = {
            'axon_diameter': axon_diameter,
            'axon_length': axon_length,
            **{key: value for key, value in all_params.items() if key.startswith('axon')}
        }
        axon_params = {'_'.join(k.split('_')[1:]): v for k, v in axon_params.items()}

        electrode_params = {
            **{key: value for key, value in all_params.items() if key.startswith('e_')}
        }
        electrode_params = {'_'.join(k.split('_')[1:]): v for k, v in electrode_params.items()}

        stimulus_params = {
            **{key: value for key, value in all_params.items() if key.startswith('s_')}
        }
        stimulus_params = {'_'.join(k.split('_')[1:]): v for k, v in stimulus_params.items()}
        stimulus_params['fun_type'] = fun_type

        axon_model = create_mrg(**axon_params)
        monitors = create_monitors(axon_model)
        electrode_params['pos'] = np.array(electrode_params['pos']).T

        if verbose:
            print('Axon params')
            print(axon_params)
            print('Stimulus params')
            print(stimulus_params)
            print('Electrode params')
            print(electrode_params)

        stimulus, pulses = create_stimulus(pos=electrode_params['pos'],
                                            rhoe=electrode_params['rhoe'],
                                            stimulus_params=stimulus_params, monitors=monitors)

        env = create_simulation(axons=axon_model,
                                stimulus=stimulus,
                                monitors=list(monitors.values()))

        env.run(sim_params['dur'] * ms, early_stopping=sim_params['early_stopping'])

        if post_processing == 'ap_block':
            y = ap_block(ap_monitor_stimulus=monitors['node_0'],
                            ap_monitor_block=monitors['node_1'],
                            start_time=2.50 * ms,
                            end_time=3.10 * ms,
                            verbose=verbose
                            )
        else:
            msg = f'Post processing method {post_processing} not implemented'
            raise NotImplementedError(msg)
        data_package = {'observations': y}
        # "time": monitors['state'].t.tolist(),
        # "v": monitors['state'].v.tolist(),
        # "pulse_a": pulses[0](monitors['state'].t).tolist(),
        # "pulse_b": pulses[1](monitors['state'].t).tolist()

        return data_package

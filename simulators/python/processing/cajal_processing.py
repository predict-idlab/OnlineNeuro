# /simulators/python/processing/cajal_processing.py
try:
    import cajal

    CAJAL_AVAILABLE = True
except ImportError:
    raise ImportError("cajal is not installed. Install it to use this module")
    CAJAL_AVAILABLE = False

__all__ = ["cajal"]

if CAJAL_AVAILABLE:
    import numpy as np
    import scipy.ndimage
    import unyt
    from cajal.nrn.monitors import StateMonitor
    from cajal.units import ms

    def nerve_block_condition(ap_monitor_left, ap_monitor_right, verbose=True):
        """
        # Blocking definition 1:
        # - It verifies if APs are present in a specific time interval (2.1 ms - 2.9 ms) on the left side.
        # - If an AP is detected on the right side, conduction is not blocked.
        # - If no APs are found on the left side during the interval, conduction is not blocked.
        # - Otherwise, conduction is considered blocked.

        Args:
            ap_monitor_left (_type_): _description_
            ap_monitor_right (_type_): _description_
            verbose (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """

        # Check if there is an AP on the left side between specific interval
        start_time_left = 2.1 * ms
        end_time_left = 2.9 * ms

        # Check the number of APs between start_time and end_time at the left side
        ap_count_left_end = ap_monitor_left.n(t=end_time_left)
        ap_count_left_start = ap_monitor_left.n(t=start_time_left)
        ap_count_left = ap_count_left_start - ap_count_left_end

        # Allow AP at the right side but not in the interval of when the stimulation pulse should arrive
        # Calculate the number of APs between start_time and end_time at the right side
        start_time_right = (
            1.4 * ms
        )  # don't allow blocking because of refractory period so set start earlier
        end_time_right = 3.8 * ms

        ap_count_right_end = ap_monitor_right.n(t=end_time_right)
        ap_count_right_start = ap_monitor_right.n(t=start_time_right)
        ap_count_right = ap_count_right_start - ap_count_right_end

        ap_times_end = ap_monitor_right.spikes()

        if verbose:
            print(
                f"Number of APs between {start_time_left} and {end_time_left} ms at the left side: {ap_count_left}"
            )
            print(f"All spike times at the right side: {ap_times_end}")

            if ap_count_right > 0:
                print("There is AP on the right side")
            elif ap_count_left == 0:
                print("NO AP at the left side")
            else:
                print("AP IS BLOCKED!!")

        if ap_count_right > 0 or ap_count_left == 0:
            return 1

        return 0

    def nerve_block_condition_2(ap_monitor_left, ap_monitor_right, verbose=True):
        """_summary_
        # Blocking definition 2:
        # - Checks for APs at the start node in the interval.
        # - If an AP is present on the right side, conduction is not blocked.
        # - If no APs are present at the left start node in the interval, conduction is not blocked.
        # - Otherwise, conduction is blocked.

        Args:
            ap_monitor_left (_type_): _description_
            ap_monitor_right (_type_): _description_
            verbose (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """

        ap_count_right = ap_monitor_right.n(t=0 * ms)

        start_time = 2.1 * ms
        end_time = 2.9 * ms

        # Calculate the number of APs between start_time and end_time at the start node
        ap_count_left_end = ap_monitor_left.n(t=end_time)
        ap_count_left_start = ap_monitor_left.n(t=start_time)
        ap_count_left = ap_count_left_start - ap_count_left_end  # Never less than zero

        ap_times_end = ap_monitor_right.spikes()

        if verbose:
            print(
                f"Number of APs between {start_time} and {end_time} ms at start node: {ap_count_left}"
            )
            print(f"All spike times at end node: {ap_times_end}")

            if ap_count_right > 0:
                print("There is AP on the right side")
            elif ap_count_left == 0:
                print("NO AP at the left side")
            else:
                print("AP IS BLOCKED!!")

        if ap_count_right > 0 or ap_count_left == 0:
            return 1

        return 0

    def nerve_block_condition_3(ap_monitor_left, ap_monitor_right, verbose=True):
        """_summary_
        # Blocking definition 3:
        #  Same as defination 2 but allow an AP on the right side before stimulation

        Args:
            ap_monitor_left (_type_): _description_
            ap_monitor_right (_type_): _description_
            verbose (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """

        # Allow AP at the right side only in the beginning
        ap_count_right = ap_monitor_right.n(t=1.1 * ms)

        # Check if there is an AP on the left side between specific interval
        start_time = 2.1 * ms
        end_time = 2.9 * ms

        # Calculate the number of APs between start_time and end_time at the start node
        ap_count_left_end = ap_monitor_left.n(t=end_time)
        ap_count_left_start = ap_monitor_left.n(t=start_time)
        ap_count_left = ap_count_left_start - ap_count_left_end  # Never less than zero

        ap_times_end = ap_monitor_right.spikes()

        if verbose:
            print(
                f"Number of APs between {start_time} and {end_time} ms at start node: {ap_count_left}"
            )
            print(f"All spike times at end node: {ap_times_end}")

            if ap_count_right > 0:
                print("There is AP on the right side")
            elif ap_count_left == 0:
                print("NO AP at the left side")
            else:
                print("AP IS BLOCKED!!")

        if ap_count_right > 0 or ap_count_left == 0:
            return 1

        return 0

    def get_ap_directional(
        binary_array, min_nodes_for_ap_start=1, y_propagation_threshold=1
    ):
        """
        Identifies APs and provides line coordinates for "upward" propagation from its
        initial min-y extent and "downward" propagation from its initial max-y extent.
        Lines are only generated if significant y-propagation occurs in that direction.
        Lines extend to the end of any plateau at the maximum y-excursion for that specific front's envelope.

        Args:
            binary_array (np.array): 2D boolean or int (0/1) array.
            min_nodes_for_ap_start (int): Minimum y-span at AP start.
            y_propagation_threshold (int): Min change in y for propagation to be considered significant.

        Returns:
            list: Dictionaries for each AP:
                - 'label', 'component_start_time', 'component_end_time' (overall component)
                - 'y_span_at_start', 'min_y_at_start', 'max_y_at_start'
                - 'upward_line_coords' (tuple of tuples or None): ((x_start, y_start_min), (x_end_up_plateau, y_extreme_of_min_envelope))
                - 'downward_line_coords' (tuple of tuples or None): ((x_start, y_start_max), (x_end_down_plateau, y_extreme_of_max_envelope))
        """
        if not isinstance(binary_array, np.ndarray) or binary_array.ndim != 2:
            raise ValueError("binary_array must be a 2D NumPy array.")

        if binary_array.dtype == bool:
            binary_array = binary_array.astype(np.int8)

        labeled_array, num_features = scipy.ndimage.label(binary_array)
        if num_features == 0:
            return []

        detected_ap_lines_data = []

        for label_idx in range(1, num_features + 1):
            coords_comp = np.argwhere(labeled_array == label_idx)
            if coords_comp.shape[0] == 0:
                continue

            t_start_comp = np.min(coords_comp[:, 1])
            t_end_comp_overall = np.max(coords_comp[:, 1])

            y_nodes_at_t_start_arr = np.unique(
                coords_comp[coords_comp[:, 1] == t_start_comp, 0]
            )
            y_span_at_start = len(y_nodes_at_t_start_arr)

            if y_span_at_start < min_nodes_for_ap_start:
                continue

            min_y_at_start = np.min(y_nodes_at_t_start_arr)
            max_y_at_start = np.max(y_nodes_at_t_start_arr)

            min_y_envelope_points = []
            max_y_envelope_points = []

            for t_current in range(t_start_comp, t_end_comp_overall + 1):
                y_nodes_at_t_current = np.unique(
                    coords_comp[coords_comp[:, 1] == t_current, 0]
                )
                if y_nodes_at_t_current.size > 0:
                    min_y_envelope_points.append(
                        (t_current, np.min(y_nodes_at_t_current))
                    )
                    max_y_envelope_points.append(
                        (t_current, np.max(y_nodes_at_t_current))
                    )

            upward_line_coords = None
            downward_line_coords = None

            # Determine Upward Line
            if min_y_envelope_points:
                # Find the most extreme y-value achieved by the min_y_envelope itself
                y_extreme_min_of_min_envelope = min(
                    y_val for t, y_val in min_y_envelope_points
                )

                propagated_upward_significantly = (
                    y_extreme_min_of_min_envelope
                    < min_y_at_start - y_propagation_threshold
                )

                if propagated_upward_significantly:
                    # Find the last time this extreme y-value was achieved by the min_y_envelope
                    t_end_plateau_up = t_start_comp
                    for t_env, y_env in reversed(min_y_envelope_points):
                        if y_env == y_extreme_min_of_min_envelope:
                            t_end_plateau_up = t_env
                            break
                    upward_line_coords = (
                        (t_start_comp, min_y_at_start),
                        (t_end_plateau_up, y_extreme_min_of_min_envelope),
                    )

            # Determine Downward Line
            if max_y_envelope_points:
                # Find the most extreme y-value achieved by the max_y_envelope itself
                y_extreme_max_of_max_envelope = max(
                    y_val for t, y_val in max_y_envelope_points
                )

                propagated_downward_significantly = (
                    y_extreme_max_of_max_envelope
                    > max_y_at_start + y_propagation_threshold
                )

                if propagated_downward_significantly:
                    # Find the last time this extreme y-value was achieved by the max_y_envelope
                    t_end_plateau_down = t_start_comp
                    for t_env, y_env in reversed(max_y_envelope_points):
                        if y_env == y_extreme_max_of_max_envelope:
                            t_end_plateau_down = t_env
                            break
                    downward_line_coords = (
                        (t_start_comp, max_y_at_start),
                        (t_end_plateau_down, y_extreme_max_of_max_envelope),
                    )

            detected_ap_lines_data.append(
                {
                    "label": label_idx,
                    "total_of_features": num_features,
                    "component_start_time": t_start_comp,
                    "component_end_time": t_end_comp_overall,
                    "upward_line_coords": upward_line_coords,
                    "downward_line_coords": downward_line_coords,
                    "y_span_at_start": y_span_at_start,
                    "min_y_at_start": min_y_at_start,
                    "max_y_at_start": max_y_at_start,
                }
            )

        detected_ap_lines_data.sort(key=lambda ap: ap["component_start_time"])
        return detected_ap_lines_data

    def detect_block(
        stim_delay: int,
        ap_directional_lines: list[dict],
        v_rec: StateMonitor,
        block_position: str = "downward",
    ):
        """Evaluates whether an AP was generated at the expeced delay/position.
        If an AP is detected, then it evaluates whether this reached the end of the fiber.
        NOTE :
        - We asume that the simulation time is sufficient for the AP to reach the end (this may not be the case)

        Args:
            stim_delay (_type_): _description_
            ap_directional_lines (_type_): _description_
            v_rec (_type_): _description_
            block_position (str, optional): _description_. Defaults to 'downward'.

        Returns:
            _type_: _description_
        """

        try:
            if isinstance(stim_delay, unyt.array.unyt_quantity):
                stim_delay = stim_delay.value
        except Exception as e:
            raise BaseException(f"stim_delay is None, this shouldn't be the case: {e}")

        init_vector = []
        key = f"{block_position}_line_coords"

        for ap_line in ap_directional_lines:
            if key in ap_line:
                if ap_line[key] is not None:
                    init_vector.append(v_rec.t[ap_line[key][0][0]])
                else:
                    init_vector.append(np.inf)
            else:
                init_vector.append(np.inf)

        # TODO, define a max limit in the difference (to make sure that the AP can be attributed correctly)
        diffs = stim_delay - np.array(init_vector)

        positive_mask = (diffs > -0.1) & (diffs < 1)  # The delay shouldn't be large
        results = {"stim_blocked": False}

        if np.any(positive_mask):
            min_pos_ix = np.where(positive_mask)[0][np.argmin(diffs[positive_mask])]
            print(f"AP detected with a difference of {diffs[positive_mask]}")
            results["stim_generated"] = True
        else:
            results["stim_generated"] = False
            results["stim_blocked"] = False
            return results

        nodes = v_rec.y
        # NOTE. Cajal doesn't propagate the AP to the last position (probably due to boundary conditions)
        if (
            block_position == "downward"
            and ap_directional_lines[min_pos_ix][key][1][1] >= len(nodes) - 2
        ):
            results["stim_blocked"] = False
        elif (
            block_position == "upward"
            and ap_directional_lines[min_pos_ix][key][1][1] <= 2
        ):
            results["stim_blocked"] = False
        else:
            results["stim_blocked"] = True

        return results

    def detect_block_naive(
        stim_delay: float,
        ap_directional_lines: list[dict],
        v_rec: StateMonitor,
        block_position: str = "downward",
    ):
        """Evaluates whether an AP was generated at the expeced delay/position.
        If an AP is detected, then it evaluates whether this reached the end of the fiber.
        NOTE :
        - We asume that the simulation time is sufficient for the AP to reach the end (this may not be the case)

        Args:
            stim_node (_type_): _description_
            ap_directional_lines (_type_): _description_
            v_rec (_type_): _description_
            block_position (str, optional): _description_. Defaults to 'downward'.

        Returns:
            _type_: _description_
        """

        results = {
            "stim_generated": False,
            "stim_blocked": False,  # Default to not blocked if not generated
            "debug_info": None,
        }

        # key = f'{block_position}_line_coords' # TODO improve this

        if block_position == "downward":
            end_position = -2
            start_position = 2
        else:
            end_position = 2
            start_position = 2

        v_binary = (v_rec.v > -20).astype(int)

        blocks = np.diff(np.concatenate(([0], v_binary[start_position], [0])))
        stim_side_aps = np.where(blocks == 1)[0]
        num_aps_gen_side = len(stim_side_aps)

        blocks = np.diff(np.concatenate(([0], v_binary[end_position], [0])))
        blockside_aps = np.where(blocks == 1)[0]
        num_aps_block_side = len(blockside_aps)

        if num_aps_gen_side > 0:
            results["stim_generated"] = True

        if num_aps_block_side < num_aps_gen_side:
            results["stim_blocked"] = True

        results["debug_info"] = {
            "num_aps_gen_side": num_aps_gen_side,
            "blockside_aps": num_aps_block_side,
        }
        return results

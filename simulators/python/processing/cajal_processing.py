# /simulators/python/processing/cajal_processing.py
import numpy as np
from cajal.nrn.monitors import APMonitor

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

    def calculate_delay(
        v_rec: StateMonitor,
        node: int,
        stim_time: float = 0.0,
        threshold: float = -20.0,
        abs_diff: bool = False,
    ):
        """Calculates the delay of the first action potential at a specific node.

        This function finds the first time the recorded voltage `v` at a given `node`
        crosses a specified `threshold`. The delay is calculated relative to a
        `stim_time`. It also counts the total number of action potentials (APs)
        detected in the signal.

        Parameters
        ----------
        v_rec : StateMonitor
            An instance of a monitor (like Brian2's StateMonitor) from which
            voltage (`.v`) and time (`.t`) arrays can be extracted.
        node : int
            The index of the node or neuron where the propagation is measured.
        stim_time : float, optional
            The time at which the stimulus was applied. The delay is calculated as
            (arrival_time - stim_time). Defaults to 0.0.
        threshold : float, optional
            The voltage threshold (in mV) that defines the arrival of an
            action potential. Defaults to -20.0.
        abs_diff : bool, optional
            If True, the absolute value of the delay is returned. Defaults to False.

        Returns
        -------
        dict
            A dictionary containing the results with the following keys:
            'delay' : float
                The calculated delay in seconds. Returns `np.inf` if the threshold
                is never crossed.
            'num_aps' : int
                The total number of action potentials (rising threshold crossings)
                detected at the node.
        """
        result = {"delay": np.inf, "num_aps": 0}

        # Ensure the node is valid to prevent IndexError
        if node >= v_rec.v.shape[0]:
            # Or raise an error, depending on desired behavior
            return result

        v = v_rec.v[node, :]
        t = v_rec.t

        # Find all indices where voltage is above threshold
        above_threshold_indices = np.where(v > threshold)[0]

        # If the threshold is never crossed, return the default values
        if len(above_threshold_indices) == 0:
            return result

        # Find the index of the first crossing
        first_arrival_idx = above_threshold_indices[0]

        # Calculate the number of APs by counting rising edges (False -> True crossings)
        as_bool = (v > threshold).astype(int)
        upward_crossings = np.diff(as_bool) == 1
        result["num_aps"] = np.sum(upward_crossings)

        # Calculate delay based on the time of the first arrival
        delay = t[first_arrival_idx] - stim_time

        if abs_diff:
            delay = np.abs(delay)

        result["delay"] = delay

        return result

    def ap_block(
        ap_monitor_stimulus: APMonitor,
        ap_monitor_block: APMonitor,
        start_time: float,
        end_time: float,
        verbose: bool = False,
    ) -> int:
        """
        Evaluate whether an action potential (AP) is blocked at a given axon node.

        Parameters
        ----------
        ap_monitor_stimulus : APMonitor
            APMonitor recording the stimulus site (start node).
            Must have methods `.n(t)` returning the number of spikes up to time t,
            and `.spikes()` returning all spike times.
        ap_monitor_block : APMonitor
            APMonitor recording the block site (end node). Must implement `.n(t)` and `.spikes()`.
        start_time : float
            Start time of the evaluation window (ms).
        end_time : float
            End time of the evaluation window (ms).
        verbose : bool, optional
            If True, prints diagnostic information (default is False).

        Returns
        -------
        int
            1 if the AP is not blocked (or no AP was generated at the stimulus site),
            0 if the AP is blocked between start and end nodes.

        Notes
        -----
        - Computes the number of APs at the stimulus node before and after the time window,
        and the number of APs at the block node at the start time.
        - Logic:
            * If there is at least one AP at the block node at start_time, AP is considered unblocked.
            * If there are zero APs at the stimulus node in the time window, return 1 (no AP).
            * Otherwise, return 0 (AP blocked).
        """
        # AP count at the block node at start_time
        ap_count_end = ap_monitor_block.n(t=start_time)

        # AP counts at the stimulus node
        ap_count_start_end = ap_monitor_stimulus.n(t=end_time)
        ap_count_start_start = ap_monitor_stimulus.n(t=start_time)

        # APs generated at stimulus node within [start_time, end_time]
        ap_count_left = max(ap_count_start_start - ap_count_start_end, 0)

        # All spike times at the block node
        ap_times_end = ap_monitor_block.spikes()

        if verbose:
            print(
                f"Number of APs between {start_time} and {end_time} ms at start node: {ap_count_left}"
            )
            print(f"All spike times at end node: {ap_times_end}")

        if ap_count_end > 0:
            if verbose:
                print("AP NOT blocked")
            return 1
        else:
            if ap_count_left == 0:
                if verbose:
                    print("NO AP at the left side")
                return 1
            else:
                if verbose:
                    print("AP is blocked")
                return 0

    def ap_block_2(
        v_monitor: APMonitor,
        threshold: float = 20.0,
        left_node: int = 15,
        right_node: int = -15,
        verbose: bool = False,
    ) -> int:
        """
        Determine if an action potential (AP) is blocked based on voltage traces.

        Parameters
        ----------
        v_monitor : APMonitor
            APMonitor object with attribute `.v` containing voltage traces.
            Expected shape: (time, nodes) or (nodes, time) depending on implementation.
        threshold : float, optional
            Voltage threshold for detecting an AP (default is 20 mV).
        left_node : int, optional
            Index of the left node to check (default is 15).
        right_node : int, optional
            Index of the right node to check (default is -15).
        verbose : bool, optional
            If True, prints which block (left or right) is detected (default is False).

        Returns
        -------
        int
            1 if a block is detected, 0 otherwise.

        Notes
        -----
        - Computes the maximum voltage at the specified nodes and compares against
        the threshold.
        - Logic:
            * `a` (Right block) is True if left_node exceeds threshold and right_node
            is below threshold.
            * `b` (Left block) is True if left_node is below threshold and right_node
            exceeds threshold.
            * Returns 1 if either `a` or `b` is True, 0 otherwise.
        """
        data = v_monitor.v
        max_time = np.max(data, axis=0)

        # Right block condition
        a = (max_time[left_node] > threshold) and (max_time[right_node] < threshold)
        # Left block condition
        b = (max_time[left_node] < threshold) and (max_time[right_node] > threshold)

        if verbose:
            if a:
                print("Right block")
            if b:
                print("Left block")
        return int(a or b)

    def nerve_block_condition(
        ap_monitor_left: StateMonitor,
        ap_monitor_right: StateMonitor,
        start_time_left: float,
        end_time_left: float,
        start_time_right: float,
        end_time_right: float,
        verbose: bool = True,
    ):
        """
        Determine whether a nerve conduction block occurred (blocking definition 1).
            - It verifies if APs are present in a specific time interval (2.1 ms - 2.9 ms) on the left side.
            - If an AP is detected on the right side, conduction is not blocked.
            - If no APs are found on the left side during the interval, conduction is not blocked.
            - Otherwise, conduction is considered blocked.

        Parameters
        ----------
        ap_monitor_left : StateMonitor
            AP monitor for the left fiber.
        ap_monitor_right : StateMonitor
            AP monitor for the right fiber.
        verbose : bool, optional
            Whether to print debug information (default True).

        Returns
        -------
        int
            0 if conduction not blocked, 1 if blocked.
        """

        # Check if there is an AP on the left side between specific interval
        start_time_left = start_time_left * ms
        end_time_left = end_time_left * ms
        # Allow AP at the right side but not in the interval of when the stimulation pulse should arrive
        # Calculate the number of APs between start_time and end_time at the right side
        # don't allow blocking because of refractory period so set start earlier
        start_time_right = start_time_right * ms
        end_time_right = end_time_right * ms

        # Check the number of APs between start_time and end_time at the left side
        ap_count_left_end = ap_monitor_left.n(t=end_time_left)
        ap_count_left_start = ap_monitor_left.n(t=start_time_left)
        ap_count_left = ap_count_left_start - ap_count_left_end

        ap_count_right_end = ap_monitor_right.n(t=end_time_right)
        ap_count_right_start = ap_monitor_right.n(t=start_time_right)
        ap_count_right = ap_count_right_start - ap_count_right_end

        ap_times_end = ap_monitor_right.spikes()

        if verbose:
            print(
                f"Left APs between {start_time_left} and {end_time_left}: {ap_count_left}"
            )
            print(f"All right AP times: {ap_times_end}")
            if ap_count_right > 0:
                print("AP detected on right side")
            elif ap_count_left == 0:
                print("No AP at left side")
            else:
                print("AP is blocked!")

        if ap_count_right > 0 or ap_count_left == 0:
            return 1

        return 0

    def get_ap_directional(
        binary_array: np.ndarray,
        min_nodes_for_ap_start: int = 1,
        y_propagation_threshold: int = 1,
    ) -> list[dict]:
        """
        Detect action potentials (APs) and calculate directional propagation lines.

        Identifies APs and provides line coordinates for "upward" propagation from its
        initial min-y extent and "downward" propagation from its initial max-y extent.
        Lines are only generated if significant y-propagation occurs in that direction.
        Lines extend to the end of any plateau at the maximum y-excursion for that specific front's envelope.

        Parameters
        ----------
        binary_array : np.ndarray
            2D array of shape (nodes, time), with 1 for AP presence and 0 otherwise.
        min_nodes_for_ap_start : int
            Minimum y-span at AP start to consider it significant (default 1).
        y_propagation_threshold : int
            Minimum y-change to consider as propagation (default 1).

        Returns
        -------
        list of dict:
            Dictionaries for each AP:
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
        stim_delay: int | unyt.array.unyt_quantity,
        ap_directional_lines: list[dict],
        v_rec: StateMonitor,
        block_position: str = "downward",
    ) -> dict[str, bool]:
        """
        Determine if an AP was generated and if conduction was blocked.

        Notes
        -----
            We asume that the simulation time is sufficient for the AP to reach the end
            (this may not be the case)

        Parameters
        ----------
        stim_delay : int or unyt_quantity
            Expected stimulation delay.
        ap_directional_lines : list of dict
            AP propagation line information.
        v_rec : StateMonitor
            Voltage recordings.
        block_position : str, optional
            'downward' or 'upward' (default 'downward').

        Returns
        -------
        dict
            Dictionary with keys:
            - 'stim_generated': bool, True if AP detected
            - 'stim_blocked': bool, True if conduction blocked
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
    ) -> dict[str, bool]:
        """
        Naive evaluation of AP generation and block status.

        Notes
        -----
            We asume that the simulation time is sufficient for the AP to reach the end
            (this may not be the case)

        Parameters
        ----------
        stim_delay : float
            Expected stimulation delay.
        ap_directional_lines : list of dict
            AP propagation line information.
        v_rec : StateMonitor
            Voltage recordings.
        block_position : str, optional
            'downward' or 'upward' (default 'downward').

        Returns
        -------
        dict
            Dictionary with keys:
            - 'stim_generated': bool
            - 'stim_blocked': bool
            - 'debug_info': dict with extra debug info

        TODO
        ----
            - Improve handling of block direction, see specific line.
        """
        results = {
            "stim_generated": False,
            "stim_blocked": False,  # Default to not blocked if not generated
            "debug_info": None,
        }

        # TODO improve this
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

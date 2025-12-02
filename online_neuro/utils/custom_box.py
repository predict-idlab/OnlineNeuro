# /online_neuro/utils/custom_box.py

from typing import Optional

from trieste.space import Box
from trieste.types import TensorType


class CustomBox(Box):
    """
    Extends trieste.space.Box to provide a unified interface for multiple
    deterministic and stochastic sampling methods.

    This class centralizes common sampling algorithms (Random, Halton, Sobol)
    and adds support for feasibility checks when sampling.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the CustomBox, passing all arguments directly to the
        parent trieste.space.Box constructor.

        See trieste.space.Box documentation for required arguments.
        """
        super().__init__(*args, **kwargs)

    def sample_method(
        self,
        num_samples: int,
        seed: Optional[int] = None,
        skip: Optional[int] = None,
        max_tries: int = 100,
        sampling_method: str = "sobol",
    ) -> TensorType:
        """
        Samples points within the search space using a specified method.

        The supported methods include standard random, Halton, and Sobol sampling,
        as well as variants that attempt to ensure samples are feasible.

        Parameters
        ----------
        num_samples : int
            The number of samples to generate.
        seed : int, optional
            The random seed used for stochastic methods ('random', 'halton')
            and feasible sampling methods based on them. Used if `sampling_method`
            is not 'sobol' or 'sobol_feasible'.
        skip : int, optional
            The number of initial Sobol sequence points to skip. Used only if
            `sampling_method` is 'sobol' or 'sobol_feasible'.
        max_tries : int, optional
            Maximum number of attempts to find feasible samples. Only used
            for methods ending in `_feasible`. Default is 100.
        sampling_method : str, optional
            The method used for sampling. Must be one of:

            - "random"
            - "halton"
            - "sobol"
            - "random_feasible"
            - "halton_feasible"
            - "sobol_feasible"

            Default is "sobol".

        Returns
        -------
        TensorType
            A tensor containing the sampled points, shape `(num_samples, dim)`.

        Raises
        ------
        ValueError
            If an unsupported `sampling_method` is provided.
        AssertionError
            If the selected sampling method fails to return any valid samples
            (e.g., if feasibility sampling fails after `max_tries`).
        """
        if sampling_method == "random":
            x = self.sample(num_samples, seed)
        elif sampling_method == "halton":
            x = self.sample_halton(num_samples, seed)
        elif sampling_method == "sobol":
            x = self.sample_sobol(num_samples, skip)
        elif sampling_method == "random_feasible":
            x = self.sample_feasible(num_samples, seed, max_tries)
        elif sampling_method == "halton_feasible":
            x = self.sample_feasible(num_samples, seed, max_tries)
        elif sampling_method == "sobol_feasible":
            x = self.sample_feasible(num_samples, skip, max_tries)
        else:
            raise ValueError(f"Unsupported sampling method: {sampling_method}")

        assert (
            x is not None
        ), f"Sampled method {sampling_method} did not return valid samples"
        return x

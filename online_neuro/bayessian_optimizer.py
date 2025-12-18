# online_neuro/bayessian_optimizer.py
#
"""
AskTellOptimizerHistory: Modified AskTell interface from Trieste

This module provides a custom extension of Trieste's :class:`AskTellOptimizer`
designed to manage and persistently store the optimization history and model
checkpoints.

The primary purpose is to:
- Guaranteed Persistence: Every time the optimizer asks for a new point,
    it creates a persistent checkpoint on disk, making the experiment recoverable.
- Explicit State Management: The self.history list and self.steps counter provide a clear,
    Pythonic way to introspect the optimization progress without relying solely on external tools.
- Simplified Workflow: The ask_and_save method bundles the three essential actions at each step
    (ask, record history, save checkpoint) into one call.

This class is crucial for resuming long-running experiments and for post-hoc
analysis of the optimization trajectory.

@Author Diego Nieves
"""

import pickle
import warnings
from collections import Counter
from pathlib import Path
from typing import Generic, Mapping, Optional, Sequence, Type, TypeVar, Union

import numpy as np
import tensorflow as tf
from trieste import logging
from trieste.acquisition.rule import AcquisitionRule
from trieste.ask_tell_optimization import AskTellOptimizer
from trieste.bayesian_optimizer import (
    FrozenRecord,
    ProbabilisticModelType,
    Record,
    StateType,
    TrainableProbabilisticModelType,
    write_summary_initial_model_fit,
    write_summary_observations,
)
from trieste.data import Dataset
from trieste.models.interfaces import TrainableProbabilisticModel
from trieste.observer import OBJECTIVE, Observer
from trieste.space import SearchSpace, SearchSpaceType
from trieste.types import State, Tag, TensorType
from trieste.utils.misc import LocalizedTag, Timer, get_value_for_tag

AskTellOptimizerType = TypeVar("AskTellOptimizerType")
AskTellOptimizerHistoryType = TypeVar("AskTellOptimizerHistoryType")


def write_summary_init(
    observer: Observer,
    search_space: SearchSpace,
    feature_names: np.ndarray | list | None,
    acquisition_rule: AcquisitionRule[
        TensorType | State[StateType | None, TensorType],
        SearchSpaceType,
        TrainableProbabilisticModelType,
    ],
    datasets: Mapping[Tag, Dataset],
    models: Mapping[Tag, TrainableProbabilisticModel],
) -> None:
    """
    Writes initial Bayesian Optimization loop metadata to the TensorBoard summary.

    This is a modified version of the standard Trieste summary writer, optimized
    for use cases where steps are tracked externally or one step at a time.

    Parameters
    ----------
    observer : Observer
        The observer used for data collection.
    search_space : SearchSpace
        The constrained search space for optimization.
    feature_names : Optional[np.ndarray | list[str]]
        A list or array of the names corresponding to the input features.
    acquisition_rule : AcquisitionRule
        The acquisition rule currently driving the optimization process.
    datasets : Mapping[Tag, Dataset]
        The initial datasets collected.
    models : Mapping[Tag, TrainableProbabilisticModel]
        The probabilistic models associated with each objective tag.
    """

    devices = tf.config.list_logical_devices()
    logging.text(
        "metadata",
        f"Observer: `{observer}`\n\n"
        f"Number of initial points: "
        f"`{dict((k, len(v)) for k, v in datasets.items())}`\n\n"
        f"Search Space: `{search_space}`\n\n"
        f"Features: `{feature_names}`\n\n"
        f"Acquisition rule:\n\n    {acquisition_rule}\n\n"
        f"Models:\n\n    {models}\n\n"
        f"Available devices: `{dict(Counter(d.device_type for d in devices))}`",
    )


class AskTellOptimizerHistory(
    AskTellOptimizer, Generic[StateType, TrainableProbabilisticModelType]
):
    """
    A subclass of Trieste's AskTellOptimizer that maintains an explicit history
    of optimization steps and provides filesystem checkpointing functionality.
    """

    def __init__(
        self,
        name: str = "default",
        # observer: str = "default",
        track_path: Optional[Path | str] = None,
        overwrite: bool = False,
        fit_model: bool = True,
        *args,
        **kwargs,
    ):
        """
        Initialize the optimizer with history tracking and checkpoint paths.

        Parameters
        ----------
        name : str, optional
            A descriptive name for the experiment, used primarily for logging. Defaults to "default".
        track_path : Optional[Path | str], optional
            Directory path where checkpoints (pickled state) will be saved.
            Defaults to the current directory if None.
        overwrite : bool, optional
            If True, checkpoint files will overwrite existing files (used for
            resuming optimization or testing). If False, the internal step counter
            is used to enforce sequential checkpoint names. Defaults to False.
        fit_model : bool, optional
            Whether to trigger model fitting immediately after initialization (based
            on the initial data provided in `*args`/`**kwargs`). Defaults to True.
        *args : Any
            Positional arguments passed to `AskTellOptimizer.__init__`.
        **kwargs : Any
            Keyword arguments passed to `AskTellOptimizer.__init__`.
        """

        kwargs.pop("name", None)
        super().__init__(fit_model=fit_model, *args, **kwargs)

        self.name = name

        self.history: list[
            FrozenRecord[StateType, TrainableProbabilisticModelType]
            | Record[StateType, TrainableProbabilisticModelType]
        ] = []
        self.steps = 0

        if track_path:
            self.track_path = Path(track_path)
        else:
            self.track_path = Path(".")

        self.overwrite = overwrite
        self._initial_fit = fit_model

    def initial_fit(self) -> None:
        assert (
            not self._initial_fit
        ), "This model was already fitted, initial_fit shouldn't be called"

        self._initial_fit = True

        with Timer() as initial_model_fitting_timer:
            for tag, model in self._models.items():
                # Prefer local dataset if available.
                tags = [tag, LocalizedTag.from_tag(tag).global_tag]
                _, dataset = get_value_for_tag(self._filtered_datasets, *tags)
                assert dataset is not None
                self.update_model(model, dataset)

        summary_writer = logging.get_tensorboard_writer()
        if summary_writer:
            with summary_writer.as_default(step=logging.get_step_number()):
                write_summary_initial_model_fit(
                    self._datasets, self._models, initial_model_fitting_timer
                )

    def tell(
        self,
        new_data: Mapping[Tag, Dataset] | Dataset,
        new_data_ixs: Sequence[TensorType] | None = None,
    ) -> None:
        """
        Update optimizer state with new data and retrains the probabilistic models.

        Note: This overrides the base `tell` method but keeps the core logic,
        ensuring consistency with history tracking.

        It keeps a simpler/cleaner structure but comes with the assumption that new_data matches the
        mapping stored (which should be the case when constructing training loops)

        Parameters
        ----------
        new_data : Mapping[Tag, Dataset] | Dataset
            New observed data.
        new_data_ixs: list[int]
            Not used, included to match Trieste's template

        Raises
        ------
        ValueError
            If keys in ``new_data`` do not match those in already built dataset.
        """
        if isinstance(new_data, Dataset):
            new_data = {OBJECTIVE: new_data}
        elif isinstance(new_data, dict):
            for k, v in new_data.items():
                if not isinstance(v, Dataset):
                    raise ValueError("Dataset received is not of type trieste.Dataset")
        else:
            # TODO eventually tf datasets could be processed here.

            raise ValueError(
                "Dataset received is not of type trieste.Dataset nor a valid dictionary"
            )
        # The datasets must have the same keys as the existing datasets. Only exception is if
        # the existing datasets are all global, in which case the dataset will be appropriately
        # updated below for the next iteration.
        datasets_indices = {
            LocalizedTag.from_tag(tag).local_index for tag in self._datasets.keys()
        }
        if self._datasets.keys() != new_data.keys() and datasets_indices != {None}:
            raise ValueError(
                f"new_data keys {new_data.keys()} doesn't "
                f"match dataset keys {self._datasets.keys()}"
            )

        for tag, new_dataset in new_data.items():
            self._datasets[tag] += new_dataset

        self._filtered_datasets = self._acquisition_rule.filter_datasets(
            self._models, self._datasets
        )

        with Timer() as model_fitting_timer:
            for tag, model in self._models.items():
                # Always use the matching dataset to the model. If the model is
                # local, then the dataset should be too by this stage.
                dataset = self._filtered_datasets[tag]
                self.update_model(model, dataset)

        summary_writer = logging.get_tensorboard_writer()
        if summary_writer:
            with summary_writer.as_default(step=logging.get_step_number()):
                write_summary_observations(
                    self._datasets,
                    self._models,
                    new_data,
                    model_fitting_timer,
                    self._observation_plot_dfs,
                )

    def ask_and_save(self, save_acq_fn=False, *args, **kwargs) -> Optional[TensorType]:
        """
        Requests the next batch of query points, records the current optimizer
        state to the internal history, and saves a checkpoint to disk.

        Parameters
        ----------
        save_acq_fn : bool, optional
            If True, the current acquisition function object is also pickled to disk.
            Defaults to False.
        *args : Any
            Positional arguments passed to `ask`.
        **kwargs : Any
            Keyword arguments passed to `ask`.

        Returns
        -------
        Optional[TensorType]
            The next batch of query points suggested by the acquisition rule,
            or None if the acquisition rule terminates the search.
        """

        query_points = self.ask(*args, **kwargs)

        if not self.overwrite:
            assert (
                len(self.history) == self.steps
            ), "The model has taken more steps than history recorded?"

        record = self.to_record()
        self.history.append(record)

        self.save(fname=f"step_{self.steps}", save_acq_fn=save_acq_fn)

        if not self.overwrite:
            self.steps += 1

        return query_points

    def save(self, fname: str = "", save_acq_fn: bool = False):
        """
        Saves the current optimization state and optionally the acquisition function
        to the filesystem defined by `self.track_path`.

        Parameters
        ----------
        fname : str, optional
            A descriptive filename suffix (e.g., "step_10"). If empty,
            uses "state_final.pickle".
        save_acq_fn : bool, optional
            If True, attempts to pickle the acquisition function object. Defaults to False.
        """

        if fname:
            state_name = f"state_{fname}.pickle"
            acq_name = f"acq_{fname}.pickle"
        else:
            state_name = "state_final.pickle"
            acq_name = "acq_final.pickle"

        state = self.to_record()
        state.save(self.track_path / state_name)

        if save_acq_fn:
            acq_fn = self._acquisition_rule.acquisition_function
            if acq_fn is not None:
                with open(self.track_path / acq_name, "wb") as f:
                    pickle.dump(acq_fn, f)

    @classmethod
    def from_record(
        cls: Type[AskTellOptimizerType],
        record: (
            Record[StateType, ProbabilisticModelType]
            | FrozenRecord[StateType, ProbabilisticModelType]
        ),
        search_space: SearchSpaceType,
        acquisition_rule: Optional[
            AcquisitionRule[
                Union[TensorType, State[Optional[StateType], TensorType]],
                SearchSpaceType,
                ProbabilisticModelType,
            ]
        ] = None,
        track_path: Path | str | None = None,
        overwrite: bool = False,
        # NOTE: Adding name and step_count for full restoration context
        name: str = "resumed_experiment",
        step_count: Optional[int] = None,
    ) -> object:
        """
        Creates a new :class:`~AskTellOptimizerHistory` instance from a previously
        saved optimization state record.

        Parameters
        ----------
        record : Record | FrozenRecord
            Optimization state record containing datasets, models, and acquisition state.
        search_space : SearchSpaceType
            The space over which to search.
        acquisition_rule : Optional[AcquisitionRule], optional
            The acquisition rule to use for future steps. If None, the default EGO
            rule would be used by the base class (though often you want to load
            the same rule used previously).
        track_path : Optional[Path | str], optional
            Path where future checkpoints should be saved.
        overwrite : bool, optional
            Whether future checkpoints should overwrite existing files.
        name : str, optional
            The name of the resumed experiment. Defaults to "resumed_experiment".
        step_count : Optional[int], optional
            The step number at which the experiment is being resumed. If provided,
            `self.steps` is set to this value (e.g., if resuming from `step_10`,
            set this to 10). If None, it defaults to calculating steps from the
            dataset length if possible, or 0.

        Returns
        -------
        AskTellOptimizerType
            A new instance of :class:`~AskTellOptimizerHistory`, initialized for resumption.
        """

        # 1. Initialize the base AskTellOptimizer state
        # We pass fit_model=False because the models in the record are already trained.

        # We use a temporary instance of AskTellOptimizer (or AskTellOptimizerHistory
        # using the base init signature) to access the base class attributes without
        # running custom history setup yet.

        instance = AskTellOptimizerHistory(  # type: ignore
            # Pass base class parameters
            search_space=search_space,
            datasets=record.datasets,
            models=record.models,
            acquisition_rule=acquisition_rule,
            acquisition_state=record.acquisition_state,
            fit_model=False,  # Essential: Do not re-fit models
            # Pass custom AskTellOptimizerHistory parameters
            name=name,
            track_path=track_path,
            overwrite=overwrite,
        )

        # 2. Set custom AskTellOptimizerHistory state parameters

        # Determine the current step count. If provided explicitly, use it.
        # Otherwise, infer it based on the number of observations (a common proxy
        # for sequential BO steps, assuming one sample per step, or use 0).
        if step_count is not None:
            resumed_step = step_count
        else:
            # Fallback: Count the number of points in the primary dataset (OBJECTIVE)
            # and subtract the initial sample size (which is usually a single step 0)
            if record.datasets and OBJECTIVE in record.datasets:
                # Assuming initial samples constitute step 0, and subsequent points are steps 1, 2, ...
                # This requires knowing the batch size, but len() is often the safest bet.
                resumed_step = len(record.datasets[OBJECTIVE])
            else:
                resumed_step = 0

        instance.steps = resumed_step

        # Optional: If required, logic to load previous history files would go here.
        # For simplicity, we start the history list with the current record.
        instance.history = [record]

        if instance.overwrite:
            warnings.warn(
                f"Optimizer resumed at step {instance.steps}. "
                "Subsequent steps will overwrite existing checkpoints due to `overwrite=True`."
            )

        return instance

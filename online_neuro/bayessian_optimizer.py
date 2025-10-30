# Copyright 2021 The Trieste Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module contains the :class:`AskTellHistory`

@Author Diego Nieves
Modified version of Trieste's AskTell.
- keep an internal state of history (as Tensorflow logging may not always be handy)
- TODO
"""

import pickle
from collections import Counter
from pathlib import Path
from typing import Generic, Mapping, Optional, Type, TypeVar

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


def write_summary_init(
    observer: Observer,
    search_space: SearchSpace,
    feature_names: Optional[np.ndarray | list],
    acquisition_rule: AcquisitionRule[
        TensorType | State[StateType | None, TensorType],
        SearchSpaceType,
        TrainableProbabilisticModelType,
    ],
    datasets: Mapping[Tag, Dataset],
    models: Mapping[Tag, TrainableProbabilisticModel],
) -> None:
    """Write initial BO loop TensorBoard summary.
    Modified version from Trieste, it doesn't save the number of steps as we assume
    1 step at a time.
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
    def __init__(
        self,
        observer: str = "default",
        track_path: Optional[Path | str] = None,
        overwrite: bool = False,
        fit_model: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(fit_model=fit_model, *args, **kwargs)
        self._observer = observer
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
        if fit_model:
            self._initial_fit = True
        else:
            self._initial_fit = False

    def initial_fit(self):
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

    def tell(self, new_data: Mapping[Tag, Dataset] | Dataset) -> None:
        """Updates optimizer state with new data.

        :param new_data: New observed data.
        :raise ValueError: If keys in ``new_data`` do not match those in already built dataset.
        """
        if isinstance(new_data, Dataset):
            new_data = {OBJECTIVE: new_data}

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

    def ask_and_save(self, save_acq_fn=False, *args, **kwargs):
        """
        Single method to request samples, but also backup state, acquisition_function
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
        acquisition_rule: (
            AcquisitionRule[
                TensorType | State[StateType | None, TensorType],
                SearchSpaceType,
                ProbabilisticModelType,
            ]
            | None
        ) = None,
        track_path: Optional[Path | str] = None,
        overwrite: bool = False,
    ) -> AskTellOptimizerType:
        """Creates new :class:`~AskTellOptimizer` instance from provided optimization state.
        Model training isn't triggered upon creation of the instance.

        :param record: Optimization state record.
        :param search_space: The space over which to search for the next query point.
        :param acquisition_rule: The acquisition rule, which defines how to search for a new point
            on each optimization step. Defaults to
            :class:`~trieste.acquisition.rule.EfficientGlobalOptimization` with default
            arguments.
        :param track_path: Path to file's location
        :param overwrite: Whether file is overwritten
        :return: New instance of :class:`~AskTellOptimizer`.
        # TODO We currently don't load the history, but do we actually need it?

        """
        # we are recovering previously saved optimization state
        # so the model was already trained
        # thus there is no need to train it again

        # type ignore below is because this relies on subclasses not overriding __init__
        # ones that do may also need to override this to get it to work

        raise NotImplementedError(" Not implemented yet")
        # return cls(  # type: ignore
        #     track_path=track_path,
        #     search_space=search_space,
        #     datasets=record.datasets,
        #     models=record.models,
        #     acquisition_rule=acquisition_rule,
        #     acquisition_state=record.acquisition_state,
        #     fit_model=False,
        # )

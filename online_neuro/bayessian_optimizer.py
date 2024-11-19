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
This module contains the :class:`BayesianOptimizer` class, used to perform Bayesian optimization.

@Author Diego Nieves
Modified version of Trieste's BayesianOptimizer.
- Remove requirements and links to ~Observer~ Class.
- Input processing (scaling) done within bo data consumption.
- Explicit optimization.
- Request of datapoints and early stopping is done in an outer loop.
- Other changes concerning what information is saved on records.


"""
import absl
from trieste.bayesian_optimizer import *
from trieste.acquisition.rule import ResultType
import warnings
from .utils import CustomMinMaxScaler, SearchSpacePipeline
import numpy as np


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
    models: Mapping[Tag, TrainableProbabilisticModel]
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


# this should be a generic NamedTuple, but mypy doesn't support them
#  https://github.com/python/mypy/issues/685
@dataclass(frozen=True)
class OptimizationResult(Generic[StateType, ProbabilisticModelType]):
    """The final result, and the historical data of the optimization process."""

    final_result: Result[Record[StateType, ProbabilisticModelType]]
    """
    The final result of the optimization process. This contains either a :class:`Record` or an
    exception.
    """

    history: list[
        Record[StateType, ProbabilisticModelType] | FrozenRecord[StateType, ProbabilisticModelType]
    ]
    r"""
    The history of the :class:`Record`\ s from each step of the optimization process. These
    :class:`Record`\ s are created at the *start* of each loop, and as such will never
    include the :attr:`final_result`. The records may be either in memory or on disk.
    """

    @staticmethod
    def step_filename(step: int) -> str:
        """Default filename for saved optimization steps."""
        return f"step.{step:0d}.pickle"

    STEP_GLOB: ClassVar[str] = "step.*.pickle"
    RESULTS_FILENAME: ClassVar[str] = "results.pickle"

    def astuple(
        self,
    ) -> tuple[
        Result[Record[StateType, ProbabilisticModelType]],
        list[
            Record[StateType, ProbabilisticModelType]
            | FrozenRecord[StateType, ProbabilisticModelType]
        ],
    ]:
        """
        **Note:** In contrast to the standard library function :func:`dataclasses.astuple`, this
        method does *not* deepcopy instance attributes.

        :return: The :attr:`final_result` and :attr:`history` as a 2-tuple.
        """
        return self.final_result, self.history

    @property
    def is_ok(self) -> bool:
        """`True` if the final result contains a :class:`Record`."""
        return self.final_result.is_ok

    @property
    def is_err(self) -> bool:
        """`True` if the final result contains an exception."""
        return self.final_result.is_err

    def try_get_final_datasets(self) -> Mapping[Tag, Dataset]:
        """
        Convenience method to attempt to get the final data.

        :return: The final data, if the optimization completed successfully.
        :raise Exception: If an exception occurred during optimization.
        """
        return self.final_result.unwrap().datasets

    def try_get_final_dataset(self) -> Dataset:
        """
        Convenience method to attempt to get the final data for a single dataset run.

        :return: The final data, if the optimization completed successfully.
        :raise Exception: If an exception occurred during optimization.
        :raise ValueError: If the optimization was not a single dataset run.
        """
        datasets = self.try_get_final_datasets()
        # Ignore local datasets.
        datasets = ignoring_local_tags(datasets)
        if len(datasets) == 1:
            return next(iter(datasets.values()))
        else:
            raise ValueError(f"Expected a single dataset, found {len(datasets)}")

    def try_get_optimal_point(self) -> tuple[TensorType, TensorType, TensorType]:
        """
        Convenience method to attempt to get the optimal point for a single dataset,
        single objective run.

        :return: Tuple of the optimal query point, observation and its index.
        """
        dataset = self.try_get_final_dataset()
        if tf.rank(dataset.observations) != 2 or dataset.observations.shape[1] != 1:
            raise ValueError("Expected a single objective")
        if tf.reduce_any(
            [
                isinstance(model, SupportsCovarianceWithTopFidelity)
                for model in self.try_get_final_models()
            ]
        ):
            raise ValueError("Expected single fidelity models")
        arg_min_idx = tf.squeeze(tf.argmin(dataset.observations, axis=0))
        return dataset.query_points[arg_min_idx], dataset.observations[arg_min_idx], arg_min_idx

    def try_get_final_models(self) -> Mapping[Tag, ProbabilisticModelType]:
        """
        Convenience method to attempt to get the final models.

        :return: The final models, if the optimization completed successfully.
        :raise Exception: If an exception occurred during optimization.
        """
        return self.final_result.unwrap().models

    def try_get_final_model(self) -> ProbabilisticModelType:
        """
        Convenience method to attempt to get the final model for a single model run.

        :return: The final model, if the optimization completed successfully.
        :raise Exception: If an exception occurred during optimization.
        :raise ValueError: If the optimization was not a single model run.
        """
        models = self.try_get_final_models()
        # Ignore local models.
        models = ignoring_local_tags(models)
        if len(models) == 1:
            return next(iter(models.values()))
        else:
            raise ValueError(f"Expected single model, found {len(models)}")

    @property
    def loaded_history(self) -> list[Record[StateType, ProbabilisticModelType]]:
        """The history of the optimization process loaded into memory."""
        return [record if isinstance(record, Record) else record.load() for record in self.history]

    def save_result(self, path: Path | str) -> None:
        """Save the final result to disk. Will overwrite any existing file at the same path."""
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, "wb") as f:
            dill.dump(self.final_result, f, dill.HIGHEST_PROTOCOL)

    def save(self, base_path: Path | str) -> None:
        """Save the optimization result to disk. Will overwrite existing files at the same path."""
        path = Path(base_path)
        num_steps = len(self.history)
        self.save_result(path / self.RESULTS_FILENAME)
        for i, record in enumerate(self.loaded_history):
            record_path = path / self.step_filename(i, num_steps)
            record.save(record_path)

    @classmethod
    def from_path(
        cls, base_path: Path | str
    ) -> OptimizationResult[StateType, ProbabilisticModelType]:
        """Load a previously saved OptimizationResult."""
        try:
            with open(Path(base_path) / cls.RESULTS_FILENAME, "rb") as f:
                result = dill.load(f)
        except FileNotFoundError as e:
            result = Err(e)

        history: list[
            Record[StateType, ProbabilisticModelType]
            | FrozenRecord[StateType, ProbabilisticModelType]
        ] = [FrozenRecord(file) for file in sorted(Path(base_path).glob(cls.STEP_GLOB))]
        return cls(result, history)


class BayesianOptimizer(Generic[SearchSpaceType]):
    """
    This class performs Bayesian optimization, the data-efficient optimization of an expensive
    black-box *objective function* over some *search space*.
    """

    def __init__(self, observer: str, search_space_pipe: SearchSpacePipeline,
                 feature_names: Optional[np.ndarray | list] = None,
                 track_state: bool = True,
                 track_path: Optional[Path | str] = None,
                 acquisition_rule: AcquisitionRule[
                                       TensorType | State[StateType | None, TensorType],
                                       SearchSpaceType,
                                       TrainableProbabilisticModelType
                                   ] |
                                   AcquisitionRule[
                                       State[StateType | None, TensorType],
                                       SearchSpaceType,
                                       TrainableProbabilisticModelType
                                   ] |
                                   AcquisitionRule[
                                       TensorType, SearchSpaceType, TrainableProbabilisticModelType
                                   ] = None,
                 acquisition_state: StateType | None = None,

                 ):
        """
        :param observer: The name of the function ~observer~ (kept just so structure is similar to Trieste's)
        :param search_space: The space over which to search. Must be a
            :class:`~trieste.space.SearchSpace`.
        :param scaler: customMinMaxScaler object or none. Whether inputs should be scaled or not.
        :param track_state: If `True`, this method saves the optimization state at the start of each
            step. Models and acquisition state are copied using `copy.deepcopy`.
        :param track_path: If set, the optimization state is saved to disk at this path,
            rather than being copied in memory.
        :param acquisition_rule: The acquisition rule, which defines how to search for a new point
            on each optimization step. Contrary to the original implementation, there's no defaulting to EGO.
        :param acquisition_state: The acquisition state to use on the first optimization step.
            This argument allows the caller to restore the optimization process from an existing
            :class:`Record`.

        **Type hints:**
            - The ``acquisition_rule`` must use the same type of
              :class:`~trieste.space.SearchSpace` as specified in :meth:`__init__`.
            - The ``acquisition_state`` must be of the type expected by the ``acquisition_rule``.
              Any acquisition state in the optimization result will also be of this type.
        """

        self._observer = observer
        self.result: OptimizationResult = None

        self._search_space_pipe = search_space_pipe
        self._search_space = search_space_pipe.search_space
        self._scaler = search_space_pipe.scaler

        self._feature_names = feature_names

        self._steps = 0

        self._crash_result = None
        self._result = None
        self._datasets = None
        self._models = None

        self._track_state = track_state
        self._track_path = track_path

        self._acquisition_rule = acquisition_rule
        self._acquisition_state = acquisition_state

        self.history: list[
            FrozenRecord[StateType, TrainableProbabilisticModelType]
            | Record[StateType, TrainableProbabilisticModelType]
        ] = []
        self.query_plot_dfs: dict[int, pd.DataFrame] = {}

        self.observation_plot_dfs = None

    def __repr__(self) -> str:
        """"""
        return f"BayesianOptimizer({self._observer!r}, {self._search_space!r})"

    def request_query_points(self,
                             datasets: Mapping[Tag, Dataset] | Dataset,
                             models: Mapping[Tag, TrainableProbabilisticModelType] | TrainableProbabilisticModelType,
                             fit_model: bool = True,
                             fit_initial_model: bool = True
                             ):
        """
        Obtain the query points based on the minimizer of the ``dataset`` and ``search_space`` (specified at
        :meth:`__init__`). This is the central implementation of the Bayesian optimization loop.

        - Finds the next points with which to query the ``observer`` using the
          ``acquisition_rule``'s :meth:`acquire` method, passing it the ``search_space``,
          ``datasets``, ``models``, and current acquisition state.

        Pass the given points to the bo-object.
        - Queries the ``observer`` *once* at those points.

        Use the optimize method to
        - Update the dataset(s) and model(s) with the data from the ``observer``.

        If any errors are raised during the request, this method will catch and return
        them instead and print a message (using `absl` at level `absl.logging.ERROR`).

        :param datasets: The known observer query points and observations for each tag.
        :param models: The model to use for each :class:`~trieste.data.Dataset` in
            ``datasets``.

        :param fit_model: If `False` then we never fit the model during BO (e.g. if we
            are using a rule that doesn't rely on the models and don't want to waste computation).
        :param fit_initial_model: If `False` then we assume that the initial models have
            already been optimized on the datasets and so do not require optimization before
            the first optimization step.
        :return: A :class:`OptimizationResult`. The :attr:`final_result` element contains either
            the final optimization data, models and acquisition state, or, if an exception was
            raised while executing the optimization loop, it contains the exception raised. In
            either case, the :attr:`history` element is the history of the data, models and
            acquisition state at the *start* of each optimization step (up to and including any step
            that fails to complete). The history will never include the final optimization result.
        :raise ValueError: If any of the following are true:
            - the keys in ``datasets`` and ``models`` do not match
            - ``datasets`` or ``models`` are empty
            - the default `acquisition_rule` is used and the tags are not `OBJECTIVE`.
        """
        # Copy the dataset, so we don't change the one provided by the user.
        datasets = copy.deepcopy(datasets)
        #new_sample = Dataset(query_points=query_points, observations=observer_output)

        if isinstance(datasets, Dataset):
            datasets = {OBJECTIVE: datasets}
        if not isinstance(models, Mapping):
            models = {OBJECTIVE: models}

        filtered_datasets = datasets
        # reassure the type checker that everything is tagged
        self._datasets = cast(Dict[Tag, Dataset], datasets)
        self._models = cast(Dict[Tag, TrainableProbabilisticModelType], models)

        # Get set of dataset and model keys, ignoring any local tag index. That is, only the
        # global tag part is considered.
        datasets_keys = {LocalizedTag.from_tag(tag).global_tag for tag in self._datasets.keys()}
        models_keys = {LocalizedTag.from_tag(tag).global_tag for tag in self._models.keys()}
        if datasets_keys != models_keys:
            raise ValueError(
                f"datasets and models should contain the same keys. Got {datasets_keys} and"
                f" {models_keys} respectively."
            )
        if not self._datasets:
            raise ValueError("dicts of datasets and models must be populated.")

        if self.observation_plot_dfs is None:
            self.observation_plot_dfs = observation_plot_init(self._datasets)

        summary_writer = logging.get_tensorboard_writer()
        if summary_writer:
            with summary_writer.as_default(step=0):
                write_summary_init(self._observer,
                                   self._search_space,
                                   self._feature_names,
                                   self._acquisition_rule,
                                   self._datasets,
                                   self._models
                                )

        self._steps += 1
        logging.set_step_number(self._steps)

        # TODO, move this check to a function, as in, evaluate before optimization starts if condition
        # has been met
        # if early_stop_callback and early_stop_callback(datasets, models, acquisition_state):
        #     tf.print("Optimization terminated early", output_stream=absl.logging.INFO)
        #     break

        try:
            if self._track_state:
                try:
                    if self._track_path is None:
                        datasets_copy = copy.deepcopy(self._datasets)
                        models_copy = copy.deepcopy(self._models)
                        acquisition_state_copy = copy.deepcopy(self._acquisition_state)
                        record = Record(datasets_copy, models_copy, acquisition_state_copy)
                        self.history.append(record)
                    else:
                        track_path = Path(self._track_path)
                        record = Record(self._datasets, self._models, self._acquisition_state)
                        file_name = OptimizationResult.step_filename(self._steps)
                        self.history.append(record.save(track_path / file_name))
                except Exception as e:
                    raise NotImplementedError(
                        "Failed to save the optimization state. Some models do not support "
                        "deep-copying or serialization and cannot be saved. "
                        "(This is particularly common for deep neural network models, though "
                        "some of the model wrappers accept a model closure as a workaround.) "
                        "For these models, the `track_state`` argument of the "
                        ":meth:`~trieste.bayesian_optimizer.BayesianOptimizer.optimize` method "
                        "should be set to `False`. This means that only the final model "
                        "will be available."
                    ) from e

            if self._steps == 1:
                #Fit can only occur while querying during the first iteration!
                
                # See explanation in AskTellOptimizer.__init__().
                if isinstance(self._acquisition_rule, LocalDatasetsAcquisitionRule):
                    self._datasets = with_local_datasets(self._datasets, self._acquisition_rule.num_local_datasets)
                    
                filtered_datasets = self._acquisition_rule.filter_datasets(self._models, self._datasets)
                if fit_model and fit_initial_model:
                    with Timer() as initial_model_fitting_timer:
                        for tag, model in self._models.items():
                            # Prefer local dataset if available.
                            tags = [tag, LocalizedTag.from_tag(tag).global_tag]
                            _, dataset = get_value_for_tag(filtered_datasets, *tags)
                            
                            assert dataset is not None
                            model.update(dataset)
                            optimize_model_and_save_result(model, dataset)

                    if summary_writer:
                        logging.set_step_number(0)
                        with summary_writer.as_default(step=0):
                            write_summary_initial_model_fit(self._datasets,
                                                            self._models,
                                                            initial_model_fitting_timer)

                    record = Record(self._datasets, self._models, self._acquisition_state)
                    result = OptimizationResult(Ok(record), self.history)
                    if self._track_state and self._track_path is not None:
                        result.save_result(Path(self._track_path) / OptimizationResult.RESULTS_FILENAME)

                    self.result = result

            with Timer() as query_point_generation_timer:
                points_or_stateful = self._acquisition_rule.acquire(self._search_space,
                                                                    self._models,
                                                                    datasets=filtered_datasets)
                if callable(points_or_stateful):
                    self._acquisition_state, query_points = points_or_stateful(self._acquisition_state)
                else:
                    query_points = points_or_stateful

            #TODO improve this
            print("Before it goes quack")
            print(query_points)
            try:
                if self._scaler:
                    qp_dict, qp_array = self._search_space_pipe.inverse_transform(query_points.numpy(), with_array=True)
                else:
                    qp_array = query_points.numpy()
                    qp_dict = qp_array
            except Exception as e:
                raise BaseException(e)

            if summary_writer:
                with summary_writer.as_default(step=self._steps):
                    write_summary_query_points(
                        self._datasets,
                        self._models,
                        self._search_space,
                        qp_array,
                        query_point_generation_timer,
                        self.query_plot_dfs,
                    )
            print("Summary written")
            return qp_dict, qp_array

        except Exception as error:  # pylint: disable=broad-except
            tf.print(
                f"\nQuerying failed at step {self._steps}, encountered error with traceback:"
                f"\n{traceback.format_exc()}"
                f"\nTerminating querying and storing the optimization history in self._crash_result. You may "
                f"be able to use the history to restart the process from a previous successful "
                f"retrieve via method retrieve_result()"
                f"optimization step.\n",
                output_stream=absl.logging.ERROR,
            )
            if isinstance(error, MemoryError):
                tf.print(
                    "\nOne possible cause of memory errors is trying to evaluate acquisition "
                    "\nfunctions over large datasets, e.g. when initializing optimizers. "
                    "\nYou may be able to word around this by splitting up the evaluation "
                    "\nusing split_acquisition_function or split_acquisition_function_calls.",
                    output_stream=absl.logging.ERROR,
                )
            result = OptimizationResult(Err(error), self.history)
            if self._track_state and self._track_path is not None:
                result.save_result(Path(self._track_path) / OptimizationResult.RESULTS_FILENAME)
            self._crash_result = result

    def retrieve_result(self) -> ResultType:
        if self._crash_result is not None:
            return self._crash_result
        else:
            msg = "Nothing to be seen here. \nResults are only stored internally if querying or optimization fail"
            warnings.warn(msg)
            return None

    def optimize_step(self, query_points, observer_output, fit_model: bool = True) -> None:
        """
        If ``track_state`` is enabled, then in addition to the final result, the history of the
        optimization process will also be returned. If ``track_path`` is also set, then
        the history and final result will be saved to disk rather than all being kept in memory.
        """
        # observer = self._observer
        # # If query_points are rank 3, then use a batched observer.
        # if tf.rank(query_points) == 3:
        #     observer = mk_batch_observer(observer)
        # observer_output = observer(query_points)

        if self._scaler:
            #TODO, check this is valid for batch samples(?)
            query_points = self._scaler.transform(query_points)

        new_sample = Dataset(query_points=query_points, observations=observer_output)

        if isinstance(new_sample, Dataset):
            new_sample = {OBJECTIVE: new_sample}

        # reassure the type checker that everything is tagged
        new_sample = cast(Dict[Tag, Dataset], new_sample)

        # Get set of dataset and model keys, ignoring any local tag index. That is, only the
        # global tag part is considered.
        summary_writer = logging.get_tensorboard_writer()

        try:
            # See explanation in AskTellOptimizer.__init__().
            if isinstance(self._acquisition_rule, LocalDatasetsAcquisitionRule):
                new_sample = with_local_datasets(new_sample, self._acquisition_rule.num_local_datasets)

            filtered_sample = self._acquisition_rule.filter_datasets(self._models, new_sample)

            if fit_model:
                with Timer() as model_fitting_timer:
                    for tag, model in self._models.items():
                        # Prefer local dataset if available.
                        tags = [tag, LocalizedTag.from_tag(tag).global_tag]
                        _, sample = get_value_for_tag(filtered_sample, *tags)
                        assert sample is not None
                        self._datasets[tag] += sample

                        model.update(self._datasets[tag])
                        optimize_model_and_save_result(model, self._datasets[tag])

                if summary_writer:
                    logging.set_step_number(0)
                    with summary_writer.as_default(step=self._steps):
                        write_summary_observations(
                            self._datasets,
                            self._models,
                            new_sample,
                            model_fitting_timer,
                            self.observation_plot_dfs,
                        )
                    logging.set_step_number(self._steps)

            tf.print("Optimization completed without errors", output_stream=absl.logging.INFO)

        except Exception as error:  # pylint: disable=broad-except
            tf.print(
                f"\nOptimization failed at step {self._steps}, encountered error with traceback:"
                f"\n{traceback.format_exc()}"
                f"\nTerminating optimization and returning the optimization history. You may "
                f"be able to use the history to restart the process from a previous successful "
                f"optimization step.\n",
                output_stream=absl.logging.ERROR,
            )
            if isinstance(error, MemoryError):
                tf.print(
                    "\nOne possible cause of memory errors is trying to evaluate acquisition "
                    "\nfunctions over large datasets, e.g. when initializing optimizers. "
                    "\nYou may be able to word around this by splitting up the evaluation "
                    "\nusing split_acquisition_function or split_acquisition_function_calls.",
                    output_stream=absl.logging.ERROR,
                )
            result = OptimizationResult(Err(error), self.history)
            if self._track_state and self._track_path is not None:
                result.save_result(Path(self._track_path) / OptimizationResult.RESULTS_FILENAME)
            self._crash_result = result

        record = Record(self._datasets, self._models, self._acquisition_state)
        result = OptimizationResult(Ok(record), self.history)
        if self._track_state and self._track_path is not None:
            result.save_result(Path(self._track_path) / OptimizationResult.RESULTS_FILENAME)

        self.result = result

    def continue_optimization(
        self,
        optimization_result: OptimizationResult[StateType, TrainableProbabilisticModelType],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        TODO needs method to first restore OptimizationResult
        Continue a previous optimization that either failed, was terminated early, or which
        you simply wish to run for more steps.

        :param num_steps: The total number of optimization steps, including any that have already
            been run.
        :param optimization_result: The optimization result from which to extract the datasets,
            models and acquisition state. If the result was successful then the final result is
            used; otherwise the last record in the history is used. The size of the history
            is used to determine how many more steps are required.
        :param args: Any more positional arguments to pass on to optimize.
        :param kwargs: Any more keyword arguments to pass on to optimize.

        Stores results in the internal param .result
        """
        history: list[
            Record[StateType, TrainableProbabilisticModelType]
            | FrozenRecord[StateType, TrainableProbabilisticModelType]
        ] = []
        history.extend(optimization_result.history)
        if optimization_result.final_result.is_ok:
            history.append(optimization_result.final_result.unwrap())
        if not history:
            raise ValueError("Cannot continue from empty optimization result")

        self.request_query_points(  # type: ignore[call-overload]
            history[-1].datasets,
            history[-1].models,
            *args,
            **kwargs,
        )

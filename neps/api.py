"""API for the neps package."""

from __future__ import annotations

import logging
import shutil
import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Concatenate, Literal

import yaml

from neps.normalization import _normalize_imported_config
from neps.optimizers import AskFunction, OptimizerChoice, OptimizerInfo, load_optimizer
from neps.runtime import _launch_runtime, _save_results
from neps.space import SearchSpace
from neps.space.neps_spaces.neps_space import (
    adjust_evaluation_pipeline_for_neps_space,
    check_neps_space_compatibility,
    convert_classic_to_neps_search_space,
    convert_neps_to_classic_search_space,
    convert_operation_to_callable,
    resolve,
)
from neps.space.neps_spaces.parameters import Operation, PipelineSpace
from neps.space.neps_spaces.string_formatter import format_value
from neps.space.parsing import convert_to_space
from neps.state import NePSState, OptimizationState, SeedSnapshot
from neps.status.status import post_run_csv
from neps.utils.common import dynamic_load_object
from neps.validation import _validate_imported_config, _validate_imported_result

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace

    from neps.optimizers.algorithms import CustomOptimizer
    from neps.state.pipeline_eval import EvaluatePipelineReturn, UserResultDict

logger = logging.getLogger(__name__)


def run(  # noqa: C901, D417, PLR0912, PLR0913, PLR0915
    evaluate_pipeline: Callable[..., EvaluatePipelineReturn] | str,
    pipeline_space: ConfigurationSpace | PipelineSpace | SearchSpace | dict | None = None,
    *,
    root_directory: str | Path = "neps_results",
    overwrite_root_directory: bool = False,
    evaluations_to_spend: int | None = None,
    max_evaluations_per_run: int | None = None,  # deprecated
    continue_until_max_evaluation_completed: bool = False,
    cost_to_spend: int | float | None = None,
    fidelities_to_spend: int | float | None = None,
    ignore_errors: bool = False,
    objective_value_on_error: float | None = None,
    cost_value_on_error: float | None = None,
    sample_batch_size: int | None = None,
    worker_id: str | None = None,
    optimizer: (
        OptimizerChoice
        | Mapping[str, Any]
        | tuple[OptimizerChoice, Mapping[str, Any]]
        | Callable[Concatenate[SearchSpace, ...], AskFunction]  # Hack, while we transit
        | Callable[Concatenate[PipelineSpace, ...], AskFunction]  # from SearchSpace to
        | Callable[Concatenate[SearchSpace | PipelineSpace, ...], AskFunction]  # Pipeline
        | CustomOptimizer
        | Literal["auto"]
    ) = "auto",
) -> None:
    """Run the optimization.

    !!! tip "Parallelization"

        To run with multiple processes or machines, execute the script that
        calls `neps.run()` multiple times. They will keep in sync using
        the file-sytem, requiring that `root_directory` be shared between them.


    ```python
    import neps
    import logging

    logging.basicConfig(level=logging.INFO)

    def evaluate_pipeline(some_parameter: float) -> float:
        validation_error = -some_parameter
        return validation_error

    class MySpace(PipelineSpace):
        dataset = "mnist"               # constant
        nlayers = neps.Integer(2,10)    # integer
        alpha = neps.Float(0.1, 1.0)    # float
        optimizer = neps.Categorical(   # categorical
            ("adam", "sgd", "rmsprop")
        )
        learning_rate = neps.Float(     # log spaced float
            lower=1e-5, upper=1, log=True
        )
        epochs =                        # fidelity integer
            neps.IntegerFidelity(1, 100)
        batch_size = neps.Integer(      # integer with a prior
            lower=32,
            upper=512,
            prior=128,
            prior_confidence="medium"
        )

    neps.run(
        evaluate_pipeline=evaluate_pipeline,
        pipeline_space=MySpace(),
        root_directory="usage_example",
        evaluations_to_spend=5,
    )
    ```

    Args:
        evaluate_pipeline: The objective function to minimize. This will be called
            with a configuration from the `pipeline_space=` that you define.

            The function should return one of the following:

            * A `float`, which is the objective value to minimize.
            * A `dict` which can have the following keys:

                ```python
                {
                    "objective_to_minimize": float,  # The thing to minimize (required)
                    "cost": float,  # The cost of the evaluate_pipeline, used by some algorithms
                    "info_dict": dict,  # Any additional information you want to store, should be YAML serializable
                }
                ```

            ??? note "`str` usage for dynamic imports"

                If a string, it should be in the format `"/path/to/:function"`.
                to specify the function to call. You may also directly provide
                an mode to import, e.g., `"my.module.something:evaluate_pipeline"`.

        pipeline_space: The pipeline space to minimize over.

            !!! tip "Optional for continuing runs"

                This parameter is **required** for the first run but **optional** when
                continuing an existing optimization. If not provided, NePS will
                automatically load the pipeline space from `root_directory/pipeline_space.pkl`.

                When provided for a continuing run, NePS will validate that it matches
                the one saved on disk to prevent inconsistencies.

            This most direct way to specify the pipeline space is as follows:

            ```python
            class MySpace(PipelineSpace):
                nlayers = neps.Integer(2,10)    # integer
                alpha = neps.Float(0.1, 1.0)    # float
                optimizer = neps.Categorical(   # categorical
                    ("adam", "sgd", "rmsprop")
                )
                learning_rate = neps.Float(     # log spaced float
                    lower=1e-5, upper=1, log=True
                )
                epochs =                        # fidelity integer
                    neps.IntegerFidelity(1, 100)
                batch_size = neps.Integer(      # integer with a prior
                    lower=32,
                    upper=512,
                    prior=128,
                    prior_confidence="medium"
                )

            neps.run(
                pipeline_space=MySpace()
            )
            ```

            You can also directly instantiate any of the parameters
            defined by [`Parameter`][neps.space.parameters.Parameter]
            and provide them directly.

            Some important properties you can set on parameters are:

            * `prior=`: If you have a good idea about what a good setting
                for a parameter may be, you can set this as the prior for
                a parameter. You specify this along with `prior_confidence`
                to assign a `"low"`, `"medium"`, or `"high"`confidence to the prior.

            !!! note "ConfigSpace support"

                You may also use a `ConfigurationSpace` object from the
                `ConfigSpace` library.

        root_directory: The directory to save progress to.

        overwrite_root_directory: If true, delete the working directory at the start of
            the run. This is, e.g., useful when debugging a evaluate_pipeline function.

        evaluations_to_spend: Number of evaluations this specific call/worker should do.
            ??? note "Limitation on Async mode"
                Currently, there is no specific number to control number of parallel evaluations running with
                the same worker, so in case you want to limit the number of parallel evaluations,
                it's crucial to limit the `evaluations_to_spend` accordingly.

        continue_until_max_evaluation_completed:
            If true, stop only after evaluations_to_spend have fully completed. In other words,
            pipelines that are still running do not count toward the stopping criterion.

        cost_to_spend: No new evaluations will start when this cost is exceeded. Requires
            returning a cost in the evaluate_pipeline function, e.g.,
            `return dict(loss=loss, cost=cost)`.

        fidelities_to_spend: accumulated fidelity spent in case of multi-fidelity after which to terminate.

        ignore_errors: Ignore hyperparameter settings that threw an error and do not raise
            an error. Error configs still count towards evaluations_to_spend.
        objective_value_on_error: Setting this and cost_value_on_error to any float will
            supress any error and will use given objective_to_minimize value instead. default: None
        cost_value_on_error: Setting this and objective_value_on_error to any float will
            supress any error and will use given cost value instead. default: None

        sample_batch_size:
            The number of samples to ask for in a single call to the optimizer.

            ??? tip "When to use this?"

                This is only useful in scenarios where you have many workers
                available, and the optimizers sample time prevents full
                worker utilization, as can happen with Bayesian optimizers.

                In this case, the currently active worker will first
                check if there are any new configurations to evaluate,
                and if not, generate `sample_batch_size` new configurations
                that the proceeding workers will then pick up and evaluate.

                We advise to only use this if:

                * You are using a `#!python "ifbo"` or `#!python "bayesian_optimization"`.
                * You have a fast to evaluate `evaluate_pipeline`
                * You have a significant amount of workers available, relative to the
                time it takes to evaluate a single configuration.

            ??? warning "Downsides of batching"

                The primary downside of batched optimization is that
                the next `sample_batch_size` configurations will not
                be able to take into account the results of any new
                evaluations, even if they were to come in relatively
                quickly.

        worker_id: An optional string to identify the worker (run instance).
            If not provided, a `worker_id` will be automatically generated using the pattern:
            `worker_<N>`, where `<N>` is a unique integer for each worker and increments with each new worker.
            A list of all workers created so far is stored in
            `root_directory/optimizer_state.pkl` under the attribute `worker_ids`.

            ??? tip "Why specify a `worker_id`?"
                Specifying a `worker_id` is useful for tracking which worker performed specific tasks
                in the results. For example, when debugging or running on a cluster, you can include
                the process ID and machine name in the `worker_id` for better traceability.

            ??? warning "Duplication of `worker_id`"
                Ensure that each worker has a unique `worker_id`. If a duplicate `worker_id` is detected,
                the optimization process will be stopped with an error to prevent overwriting the results
                of other workers.

        optimizer: Which optimizer to use.

            Not sure which to use? Leave this at `"auto"` and neps will
            choose the optimizer based on the pipeline space given.

            ??? note "Available optimizers"

                See the [optimizers documentation](../../reference/search_algorithms/landing_page_algo.md) for a list of available optimizers.

            With any optimizer choice, you also may provide some additional parameters to the optimizers.
            We do not recommend this unless you are familiar with the optimizer you are using. You
            may also specify an optimizer as a dictionary for supporting reading in serialized yaml
            formats:

            ```python
            neps.run(
                ...,
                optimizer=("priorband",
                    {
                        "sample_prior_first": True,
                    }
                )
            )
            ```

            ??? tip "Own optimzier"

                Lastly, you may also provide your own optimizer which must satisfy
                the [`AskFunction`][neps.optimizers.optimizer.AskFunction] signature.

                ```python
                class MyOpt:

                    def __init__(self, space: SearchSpace):
                        ...

                    def __call__(
                        self,
                        trials: Mapping[str, Trial],
                        budget_info: BudgetInfo | None,
                        n: int | None = None,
                    ) -> SampledConfig | list[SampledConfig]:
                        # Sample a new configuration.
                        #
                        # Args:
                        #   trials: All of the trials that are known about.
                        #   budget_info: information about the budget constraints.
                        #
                        # Returns:
                        #   The sampled configuration(s)


                neps.run(
                    ...,
                    optimizer=MyOpt,
                )
                ```

                This is mainly meant for internal development but allows you to use the NePS
                runtime to run your optimizer.

    """  # noqa: E501
    if max_evaluations_per_run is not None:
        raise ValueError(
            "`max_evaluations_per_run` is deprecated, please use "
            "`evaluations_to_spend` for limiting the number of evaluations for this run.",
        )

    # If the pipeline_space is a SearchSpace, convert it to a PipelineSpace and throw a
    # deprecation warning
    if isinstance(pipeline_space, SearchSpace | dict):
        if isinstance(pipeline_space, dict):
            pipeline_space = SearchSpace(pipeline_space)
        pipeline_space = convert_classic_to_neps_search_space(pipeline_space)
        space_lines = str(pipeline_space).split("\n")
        space_def = space_lines[1] if len(space_lines) > 1 else str(pipeline_space)
        warnings.warn(
            "Passing a SearchSpace or dictionary to neps.run is deprecated and will be"
            " removed in a future version. Please pass a PipelineSpace instead, as"
            " described in the NePS-Spaces documentation."
            " This specific space should be given as:\n\n```python\nclass"
            f" MySpace(PipelineSpace):\n{space_def}\n```\n",
            DeprecationWarning,
            stacklevel=2,
        )

    # Try to load pipeline_space from disk if not provided
    if pipeline_space is None:
        root_path = Path(root_directory)
        if root_path.exists() and not overwrite_root_directory:
            try:
                pipeline_space = load_pipeline_space(root_path)
                logger.info(
                    "Loaded pipeline space from disk. Continuing optimization with "
                    f"existing pipeline space from {root_path}"
                )
            except (FileNotFoundError, ValueError) as e:
                # If loading fails, we'll error below
                logger.debug(f"Could not load pipeline space from disk: {e}")

        # If still None, raise error
        if pipeline_space is None:
            raise ValueError(
                "pipeline_space is required for the first run. For continuing an"
                " existing run, the pipeline space will be loaded from disk. No existing"
                f" pipeline space found at: {root_directory}"
            )
    controling_params = {
        "evaluations_to_spend": evaluations_to_spend,
        "cost_to_spend": cost_to_spend,
        "fidelities_to_spend": fidelities_to_spend,
    }
    if all(x is None for x in controling_params.values()):
        warnings.warn(
            "None of the following were set, this will run idefinitely until the worker"
            " process is stopped."
            f"{', '.join(list(controling_params.keys()))}.",
            UserWarning,
            stacklevel=2,
        )

    logger.info(f"Starting neps.run using root directory {root_directory}")

    # Check if we're continuing an existing run and should load the optimizer from disk
    root_path = Path(root_directory)
    optimizer_info_path = root_path / "optimizer_info.yaml"
    is_continuing_run = optimizer_info_path.exists() and not overwrite_root_directory

    # If continuing a run and optimizer is "auto" (default), load existing optimizer
    # with its parameters
    if is_continuing_run and optimizer == "auto":
        try:
            existing_optimizer_info = load_optimizer_info(root_path)
            logger.info(
                "Continuing optimization with existing optimizer: "
                f"{existing_optimizer_info['name']}"
            )
            # Use the existing optimizer with its original parameters
            optimizer = (
                existing_optimizer_info["name"],
                existing_optimizer_info["info"],
            )  # type: ignore
        except (FileNotFoundError, KeyError) as e:
            # No existing optimizer found or invalid format, proceed with auto
            logger.debug(f"Could not load existing optimizer info: {e}")

    # Check if the pipeline_space only contains basic HPO parameters.
    # If yes, we convert it to a classic SearchSpace, to use with the old optimizers.
    # If no, we use adjust_evaluation_pipeline_for_neps_space to convert the
    # pipeline_space and only use the new NEPS optimizers.

    # If the optimizer is not a NEPS algorithm, we try to convert the pipeline_space

    neps_classic_space_compatibility = check_neps_space_compatibility(optimizer)
    if neps_classic_space_compatibility in ["both", "classic"] and isinstance(
        pipeline_space, PipelineSpace
    ):
        converted_space = convert_neps_to_classic_search_space(pipeline_space)
        if converted_space:
            pipeline_space = converted_space
    space = convert_to_space(pipeline_space)

    if neps_classic_space_compatibility == "neps" and not isinstance(
        space, PipelineSpace
    ):
        space = convert_classic_to_neps_search_space(space)

    # Optimizer check, if the search space is a Pipeline and the optimizer is not a NEPS
    # algorithm, we raise an error, as the optimizer is not compatible.
    if isinstance(space, PipelineSpace) and neps_classic_space_compatibility == "classic":
        raise ValueError(
            "The provided optimizer is not compatible with this complex search space. "
            "Please use one that is, such as 'random_search', 'hyperband', "
            "'priorband', or 'complex_random_search'."
        )

    # Log the search space after conversion
    logger.info(str(space))

    _optimizer_ask, _optimizer_info = load_optimizer(optimizer=optimizer, space=space)

    multi_fidelity_optimizers = {
        "successive_halving",
        "asha",
        "hyperband",
        "async_hb",
        "ifbo",
        "priorband",
        "moasha",
        "mo_hyperband",
        "primo",
        "neps_priorband",
        "neps_bracket_optimizer",
        "neps_hyperband",
    }

    is_multi_fidelity = _optimizer_info["name"] in multi_fidelity_optimizers

    if not is_multi_fidelity and fidelities_to_spend is not None:
        raise ValueError(
            "`fidelities_to_spend` is not allowed for non-multi-fidelity optimizers."
        )

    _eval: Callable
    if isinstance(evaluate_pipeline, str):
        module, funcname = evaluate_pipeline.rsplit(":", 1)
        eval_pipeline = dynamic_load_object(module, funcname)
        if not callable(eval_pipeline):
            raise ValueError(
                f"'{funcname}' in module '{module}' is not a callable function."
            )
        _eval = eval_pipeline
    elif callable(evaluate_pipeline):
        _eval = evaluate_pipeline
    else:
        raise ValueError(
            "evaluate_pipeline must be a callable or a string in the format"
            "'module:function'."
        )
    if isinstance(space, PipelineSpace):
        _eval = adjust_evaluation_pipeline_for_neps_space(_eval, space)

    _launch_runtime(
        evaluation_fn=_eval,  # type: ignore
        optimizer=_optimizer_ask,
        optimizer_info=_optimizer_info,
        cost_to_spend=cost_to_spend,
        fidelities_to_spend=fidelities_to_spend,
        optimization_dir=Path(root_directory),
        evaluations_to_spend=evaluations_to_spend,
        continue_until_max_evaluation_completed=continue_until_max_evaluation_completed,
        objective_value_on_error=objective_value_on_error,
        cost_value_on_error=cost_value_on_error,
        ignore_errors=ignore_errors,
        overwrite_optimization_dir=overwrite_root_directory,
        sample_batch_size=sample_batch_size,
        worker_id=worker_id,
        pipeline_space=pipeline_space,
    )

    post_run_csv(root_directory)
    root_directory = Path(root_directory)
    summary_dir = root_directory / "summary"
    logger.info(
        "The summary folder has been created, which contains csv and txt files with"
        "the output of all data in the run (short.csv - only the best; full.csv - "
        "all runs; best_config_trajectory.txt for incumbent trajectory; and "
        "best_config.txt for final incumbent)."
        f"\nYou can find summary folder at: {summary_dir}."
    )


def save_pipeline_results(
    user_result: dict,
    pipeline_id: str,
    root_directory: Path,
) -> None:
    """Persist the outcome of one pipeline evaluation.

    Args:
        user_result (dict): Dictionary returned by evaluate_pipeline. Must
            contain keys required by _save_results (e.g. "cost",
            "objective_to_minimize", optional "learning_curve", "exception",
            "info_dict").
        pipeline_id (str): Unique identifier of the pipeline/trial whose
            result is being stored. Used to locate the corresponding
            neps.core.trial.Trial object inside the optimisation state.
        root_directory (Path): Root directory of the NePS run (contains
            optimizer_info.yaml and configs/ folder).

    """
    _save_results(
        user_result=user_result,
        trial_id=pipeline_id,
        root_directory=root_directory,
    )

    full_frame_path, short_path = post_run_csv(root_directory)
    logger.info(
        "The post run summary has been created, which is a csv file with the "
        "output of all data in the run."
        f"\nYou can find a full dataframe at: {full_frame_path}."
        f"\nYou can find a quick summary at: {short_path}."
    )


def import_trials(  # noqa: C901
    evaluated_trials: Sequence[tuple[Mapping[str, Any], UserResultDict],],
    root_directory: Path | str,
    pipeline_space: SearchSpace | dict | PipelineSpace | None = None,
    overwrite_root_directory: bool = False,  # noqa: FBT001, FBT002
    optimizer: (
        OptimizerChoice
        | Mapping[str, Any]
        | tuple[OptimizerChoice, Mapping[str, Any]]
        | Callable[Concatenate[SearchSpace, ...], AskFunction]  # Hack, while we transit
        | Callable[Concatenate[PipelineSpace, ...], AskFunction]  # from SearchSpace to
        | Callable[Concatenate[SearchSpace | PipelineSpace, ...], AskFunction]  # Pipeline
        | CustomOptimizer
        | Literal["auto"]
    ) = "auto",
) -> None:
    """Import externally evaluated trials into the optimization state.

    This function allows you to add trials that have already been
    evaluated outside of NePS into the current optimization run.
    It validates and normalizes the provided configurations,
    removes duplicates, and updates the optimization state accordingly.

    Args:
        evaluated_trials (Sequence[tuple[Mapping[str, Any], UserResultDict]]):
            A sequence of tuples, each containing a configuration dictionary
            and its corresponding result.
        root_directory (Path or str): The root directory of the NePS run.
        pipeline_space (SearchSpace | dict | PipelineSpace | None): The pipeline space
            used for the optimization. If None, will attempt to load from the
            root_directory. If provided and a pipeline space exists on disk, they
            will be validated to match.
        overwrite_root_directory (bool, optional): If True, overwrite the existing
            root directory. Defaults to False.
        optimizer: The optimizer to use for importing trials.
            Can be a string, mapping, tuple, callable, or CustomOptimizer.
            Defaults to "auto".

    Returns:
        None

    Raises:
        ValueError: If any configuration or result is invalid, or if pipeline_space
            cannot be determined (neither provided nor found on disk).
        FileNotFoundError: If the root directory does not exist.
        Exception: For unexpected errors during trial import.

    Example:
        >>> import neps
        >>> from neps.state.pipeline_eval import UserResultDict
        >>> pipeline_space = neps.SearchSpace({...})
        >>> evaluated_trials = [
        ...     ({"param1": 0.5, "param2": 10},
        ...     UserResultDict(objective_to_minimize=-5.0)),
        ... ]
        >>> neps.import_trials(evaluated_trials, "my_results", pipeline_space)
    """
    if isinstance(root_directory, str):
        root_directory = Path(root_directory)

    # Try to load pipeline_space from disk if not provided
    if pipeline_space is None:
        if root_directory.exists() and not overwrite_root_directory:
            try:
                pipeline_space = load_pipeline_space(root_directory)
                logger.info(
                    "Loaded pipeline space from disk. Importing trials with "
                    f"existing pipeline space from {root_directory}"
                )
            except (FileNotFoundError, ValueError) as e:
                # If loading fails, we'll error below
                logger.debug(f"Could not load pipeline space from disk: {e}")

        # If still None, raise error
        if pipeline_space is None:
            raise ValueError(
                "pipeline_space is required when importing trials to a new run. "
                "For importing to an existing run, the pipeline space will be loaded "
                f"from disk. No existing pipeline space found at: {root_directory}"
            )
    # Note: If pipeline_space is provided, it will be validated against the one on disk
    # by NePSState.create_or_load() after necessary conversions are applied

    neps_classic_space_compatibility = check_neps_space_compatibility(optimizer)
    if neps_classic_space_compatibility in ["both", "classic"] and isinstance(
        pipeline_space, PipelineSpace
    ):
        converted_space = convert_neps_to_classic_search_space(pipeline_space)
        if converted_space:
            pipeline_space = converted_space
    space = convert_to_space(pipeline_space)

    if neps_classic_space_compatibility == "neps" and not isinstance(
        space, PipelineSpace
    ):
        space = convert_classic_to_neps_search_space(space)

    # Optimizer check, if the pipeline space is a Pipeline and the optimizer is not a NEPS
    # algorithm, we raise an error, as the optimizer is not compatible.
    if isinstance(space, PipelineSpace) and neps_classic_space_compatibility == "classic":
        raise ValueError(
            "The provided optimizer is not compatible with this complex pipeline space. "
            "Please use one that is, such as 'random_search', 'hyperband', "
            "'priorband', or 'complex_random_search'."
        )

    optimizer_ask, optimizer_info = load_optimizer(optimizer, space)

    if overwrite_root_directory and root_directory.exists():
        logger.info(
            f"Overwriting root directory '{root_directory}' as"
            " `overwrite_root_directory=True`."
        )
        shutil.rmtree(root_directory)

    state = NePSState.create_or_load(
        root_directory,
        optimizer_info=optimizer_info,
        optimizer_state=OptimizationState(
            budget=None, seed_snapshot=SeedSnapshot.new_capture(), shared_state={}
        ),
        pipeline_space=space,
    )

    normalized_trials = []
    for config, result in evaluated_trials:
        _validate_imported_config(space, config)
        _validate_imported_result(result)
        normalized_config = _normalize_imported_config(space, config)
        normalized_trials.append((normalized_config, result))

    with state._trial_lock.lock():
        state_trials = state._trial_repo.latest()
        # remove duplicates
        existing_configs = [
            tuple(sorted(t.config.items())) for t in state_trials.values()
        ]
        num_before_dedup = len(normalized_trials)
        normalized_trials = [
            t
            for t in normalized_trials
            if tuple(sorted(t[0].items())) not in existing_configs
        ]
        num_duplicates = num_before_dedup - len(normalized_trials)
        if num_duplicates > 0:
            logger.info(
                f"Skipped {num_duplicates} duplicate trial(s) (already exist in state)."
            )

        imported_trials = optimizer_ask.import_trials(
            external_evaluations=normalized_trials,
            trials=state_trials,
        )
    # create Trial objects and add to state
    state.lock_and_import_trials(imported_trials, worker_id="external")


def create_config(  # noqa: C901
    pipeline_space: PipelineSpace | None = None,
    root_directory: Path | str | None = None,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Create a configuration by prompting the user for input.

    Args:
        pipeline_space: The pipeline space to create a configuration for.
            If None, will attempt to load from
            `root_directory/pipeline_space.pkl` if `root_directory` is
            provided.
        root_directory: The root directory to load the pipeline space from
            if `pipeline_space` is None.

    Returns:
        A tuple containing the created configuration dictionary and the
        sampled pipeline.
    """
    from neps.space.neps_spaces.neps_space import NepsCompatConverter
    from neps.space.neps_spaces.sampling import IOSampler

    # Try to load pipeline_space from disk if path is provided
    if root_directory:
        try:
            loaded_space = load_pipeline_space(root_directory)
        except (FileNotFoundError, ValueError) as e:
            # If loading fails, we'll error below
            raise ValueError(
                f"Could not load pipeline space from disk at {root_directory}: {e}"
            ) from e
        # Validate loaded space is a PipelineSpace
        if not isinstance(loaded_space, PipelineSpace):
            raise ValueError(
                "create_config only supports PipelineSpace. The loaded space "
                f"from {root_directory} is not a PipelineSpace."
            )

        if pipeline_space is None:
            pipeline_space = loaded_space
        else:
            # Validate provided pipeline_space is a PipelineSpace
            if not isinstance(pipeline_space, PipelineSpace):
                raise ValueError(
                    "create_config only supports PipelineSpace. The provided "
                    "pipeline_space is not a PipelineSpace."
                )

            # Validate provided pipeline_space matches loaded one
            import pickle

            if pickle.dumps(loaded_space) != pickle.dumps(pipeline_space):
                raise ValueError(
                    "The pipeline_space provided does not match the one saved on"
                    " disk.\nPipeline space location:"
                    f" {Path(root_directory) / 'pipeline_space.pkl'}\nPlease either:\n"
                    "  1. Don't provide pipeline_space (it will be loaded automatically),"
                    " or\n  2. Provide the same pipeline_space that was used in"
                    " neps.run()"
                )
    elif pipeline_space is None:
        raise ValueError(
            "pipeline_space or root_directory is required when creating a configuration."
        )

    resolved_pipeline, resolution_context = resolve(
        pipeline_space, domain_sampler=IOSampler()
    )

    # Print the resolved pipeline

    pipeline_dict = dict(**resolved_pipeline.get_attrs())

    for name, value in pipeline_dict.items():
        if isinstance(value, Operation):
            # If the operator is a not a string, we convert it to a callable.
            if isinstance(value.operator, str):
                pipeline_dict[name] = format_value(value)
            else:
                pipeline_dict[name] = convert_operation_to_callable(value)

    return NepsCompatConverter.to_neps_config(resolution_context), pipeline_dict


def load_config(  # noqa: C901, PLR0912, PLR0915
    config_path: Path | str,
    pipeline_space: PipelineSpace | SearchSpace | None = None,
    config_id: str | None = None,
) -> dict[str, Any]:
    """Load a configuration from a neps config file.

    Args:
        config_path: Path to the neps config file.
        pipeline_space: The pipeline space used to generate the configuration.
            If None, will attempt to load from the NePSState directory.
        config_id: Optional config id to load, when only giving results folder.

    Returns:
        The loaded configuration as a dictionary.

    Raises:
        ValueError: If pipeline_space is not provided and cannot be loaded from disk.
    """
    from neps.space.neps_spaces.neps_space import NepsCompatConverter
    from neps.space.neps_spaces.sampling import OnlyPredefinedValuesSampler

    # Try to load pipeline_space from NePSState if not provided
    state = None  # Track state for later use in config loading

    if pipeline_space is None:
        try:
            # Extract the root directory from config_path
            str_path_temp = str(config_path)
            if "/configs/" in str_path_temp or "\\configs\\" in str_path_temp:
                root_dir = Path(
                    str_path_temp.split("/configs/")[0].split("\\configs\\")[0]
                )
            # If no /configs/ in path, assume it's either:
            # 1. The root directory itself
            # 2. A direct config file path (ends with .yaml/.yml)
            elif str_path_temp.endswith((".yaml", ".yml")):
                # It's a direct config file path, go up two levels
                root_dir = Path(str_path_temp).parent.parent
            else:
                # It's the root directory itself
                root_dir = Path(str_path_temp)

            state = NePSState.create_or_load(path=root_dir, load_only=True)
            pipeline_space = state.lock_and_get_search_space()

            if pipeline_space is None:
                raise ValueError(
                    "Could not load pipeline_space from disk. "
                    "Please provide pipeline_space argument or ensure "
                    "the NePSState was created with search_space saved."
                )
        except Exception as e:
            raise ValueError(
                f"pipeline_space not provided and could not be loaded from disk: {e}"
            ) from e
    else:
        # User provided a pipeline_space - validate it matches the one on disk
        from neps.exceptions import NePSError

        try:
            str_path_temp = str(config_path)
            if "/configs/" in str_path_temp or "\\configs\\" in str_path_temp:
                root_dir = Path(
                    str_path_temp.split("/configs/")[0].split("\\configs\\")[0]
                )
            # If no /configs/ in path, assume it's either:
            # 1. The root directory itself
            # 2. A direct config file path (ends with .yaml/.yml)
            elif str_path_temp.endswith((".yaml", ".yml")):
                # It's a direct config file path, go up two levels
                root_dir = Path(str_path_temp).parent.parent
            else:
                # It's the root directory itself
                root_dir = Path(str_path_temp)

            state = NePSState.create_or_load(path=root_dir, load_only=True)
            disk_space = state.lock_and_get_search_space()

            if disk_space is not None:
                # Validate that provided space matches disk space
                import pickle

                if pickle.dumps(disk_space) != pickle.dumps(pipeline_space):
                    raise NePSError(
                        "The pipeline_space provided does not match the one saved on"
                        " disk.\\nPipeline space location:"
                        f" {root_dir / 'pipeline_space.pkl'}\\nPlease either:\\n  1."
                        " Don't provide pipeline_space (it will be loaded"
                        " automatically), or\\n  2. Provide the same pipeline_space that"
                        " was used in neps.run()"
                    )
        except NePSError:
            raise
        except Exception:  # noqa: S110, BLE001
            # If we can't load/validate, just continue with provided space
            pass

    # Determine config_id from path
    str_path = str(config_path)
    trial_id = None

    if not str_path.endswith(".yaml") and not str_path.endswith(".yml"):
        if str_path.removesuffix("/").split("/")[-1].startswith("config_"):
            # Extract trial_id from path like "configs/config_1"
            # or "configs/config_1_rung_0"
            trial_id = str_path.removesuffix("/").split("/")[-1]
        else:
            if config_id is None:
                raise ValueError(
                    "When providing a results folder, you must also provide a config_id."
                )
            trial_id = config_id
    else:
        # Extract trial_id from yaml path like "configs/config_1/config.yaml"
        path_parts = str_path.replace("\\", "/").split("/")
        for i, part in enumerate(path_parts):
            if part == "configs" and i + 1 < len(path_parts):
                trial_id = path_parts[i + 1]
                break

    # Use the locked method from NePSState to safely read the trial
    if trial_id is not None and state is not None:
        try:
            trial = state.lock_and_get_trial_by_id(trial_id)
            config_dict = dict(trial.config)  # Convert Mapping to dict
        except Exception:  # noqa: BLE001
            # Fallback to direct file read if trial can't be loaded
            str_path_fallback = str(config_path)
            if not str_path_fallback.endswith(".yaml") and not str_path_fallback.endswith(
                ".yml"
            ):
                str_path_fallback += "/config.yaml"
            config_path = Path(str_path_fallback)
            with config_path.open("r") as f:
                config_dict = yaml.load(f, Loader=yaml.SafeLoader)
    else:
        # Fallback to direct file read
        str_path_fallback = str(config_path)
        if not str_path_fallback.endswith(".yaml") and not str_path_fallback.endswith(
            ".yml"
        ):
            str_path_fallback += "/config.yaml"
        config_path = Path(str_path_fallback)
        with config_path.open("r") as f:
            config_dict = yaml.load(f, Loader=yaml.SafeLoader)

    # Handle different pipeline space types
    if not isinstance(pipeline_space, PipelineSpace):
        # For SearchSpace (classic), just return the config dict
        return dict(config_dict) if isinstance(config_dict, Mapping) else config_dict

    # For PipelineSpace, resolve it
    converted_dict = NepsCompatConverter.from_neps_config(config_dict)

    pipeline, _ = resolve(
        pipeline_space,
        domain_sampler=OnlyPredefinedValuesSampler(converted_dict.predefined_samplings),
        environment_values=converted_dict.environment_values,
    )

    # Print the resolved pipeline

    pipeline_dict = dict(**pipeline.get_attrs())

    for name, value in pipeline_dict.items():
        if isinstance(value, Operation):
            # If the operator is a not a string, we convert it to a callable.
            if isinstance(value.operator, str):
                pipeline_dict[name] = format_value(value)
            else:
                pipeline_dict[name] = convert_operation_to_callable(value)

    return pipeline_dict


def load_pipeline_space(
    root_directory: str | Path,
) -> PipelineSpace | SearchSpace:
    """Load the pipeline space from a neps run directory.

    This is a convenience function that loads the pipeline space that was saved
    during a neps.run() call. The pipeline space is automatically saved to disk
    and can be loaded to inspect it or use it with other neps utilities.

    Args:
        root_directory: Path to the neps results directory (the same path
            that was passed to neps.run()).

    Returns:
        The pipeline space that was used in the neps run.

    Raises:
        FileNotFoundError: If no neps state is found at the given path.
        ValueError: If no pipeline space was saved in the neps run.

    Example:
        ```python
        # After running neps
        neps.run(
            evaluate_pipeline=my_function,
            pipeline_space=MySpace(),
            root_directory="results",
        )

        # Later, load the space
        space = neps.load_pipeline_space("results")
        ```
    """
    from neps.state import NePSState

    root_directory = Path(root_directory)

    try:
        state = NePSState.create_or_load(path=root_directory, load_only=True)
        pipeline_space = state.lock_and_get_search_space()

        if pipeline_space is None:
            raise ValueError(
                f"No pipeline space was saved in the neps run at: {root_directory}\n"
                "This can happen if the run was created before pipeline space "
                "persistence was added, or if the pipeline_space.pkl file was deleted."
            )

        return pipeline_space
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"No neps state found at: {root_directory}\n"
            "Please provide a valid neps results directory."
        ) from e


def load_optimizer_info(
    root_directory: str | Path,
) -> OptimizerInfo:
    """Load the optimizer information from a neps run directory.

    This function loads the optimizer metadata that was saved during a neps.run()
    call, including the optimizer name and its configuration parameters. This is
    useful for inspecting what optimizer was used and with what settings.

    Args:
        root_directory: Path to the neps results directory (the same path
            that was passed to neps.run()).

    Returns:
        A dictionary containing:
            - 'name': The name of the optimizer (e.g., 'bayesian_optimization')
            - 'info': Additional optimizer configuration (e.g., initialization kwargs)

    Raises:
        FileNotFoundError: If no neps state is found at the given path.

    Example:
        ```python
        # After running neps
        neps.run(
            evaluate_pipeline=my_function,
            pipeline_space=MySpace(),
            root_directory="results",
            optimizer="bayesian_optimization",
        )

        # Later, check what optimizer was used
        optimizer_info = neps.load_optimizer_info("results")
        print(f"Optimizer: {optimizer_info['name']}")
        print(f"Config: {optimizer_info['info']}")
        ```
    """
    from neps.state import NePSState

    root_directory = Path(root_directory)

    try:
        state = NePSState.create_or_load(path=root_directory, load_only=True)
        return state.lock_and_get_optimizer_info()
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"No neps state found at: {root_directory}\n"
            "Please provide a valid neps results directory."
        ) from e


__all__ = [
    "create_config",
    "import_trials",
    "load_config",
    "load_optimizer_info",
    "load_pipeline_space",
    "run",
    "save_pipeline_results",
]

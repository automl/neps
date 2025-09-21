"""API for the neps package."""

from __future__ import annotations

import logging
import os
import shutil
import socket
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Concatenate, Literal

import neps
import neps.optimizers.algorithms
import neps.optimizers.neps_bracket_optimizer
from neps.optimizers import AskFunction, OptimizerChoice, load_optimizer
from neps.optimizers.ask_and_tell import AskAndTell
from neps.runtime import _launch_runtime, _save_results
from neps.space.neps_spaces import neps_space
from neps.space.neps_spaces.neps_space import (
    NepsCompatConverter,
    adjust_evaluation_pipeline_for_neps_space,
    check_neps_space_compatibility,
    convert_classic_to_neps_search_space,
    convert_neps_to_classic_search_space,
)
from neps.space.neps_spaces.parameters import PipelineSpace
from neps.space.parsing import convert_to_space
from neps.state import NePSState, OptimizationState, SeedSnapshot
from neps.state.neps_state import TrialRepo
from neps.state.pipeline_eval import EvaluatePipelineReturn
from neps.status.status import post_run_csv, trajectory_of_improvements
from neps.utils.common import dynamic_load_object

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace

    from neps.optimizers.algorithms import CustomOptimizer
    from neps.space import SearchSpace
    from neps.state import EvaluatePipelineReturn

logger = logging.getLogger(__name__)


def run(  # noqa: C901, D417, PLR0913, PLR0912, PLR0915
    evaluate_pipeline: Callable[..., EvaluatePipelineReturn] | str,
    pipeline_space: ConfigurationSpace | PipelineSpace,
    *,
    root_directory: str | Path = "neps_results",
    overwrite_root_directory: bool = False,
    evaluations_to_spend: int | None = None,
    write_summary_to_disk: bool = True,
    max_evaluations_total: int | None = None,
    max_evaluations_per_run: int | None = None,
    continue_until_max_evaluation_completed: bool = False,
    cost_to_spend: int | float | None = None,
    max_cost_total: int | float | None = None,
    fidelities_to_spend: int | None = None,
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
    warmstart_configs: (
        list[
            tuple[
                dict[str, Any] | Mapping[str, Any],
                dict[str, Any] | Mapping[str, Any],
                Any,
            ]
        ]
        | None
    ) = None,
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

    MySpace(PipelineSpace):
        dataset = "mnist"               # constant
        nlayers = neps.Integer(2,10)    # integer
        alpha = neps.Float(0.1, 1.0)    # float
        optimizer = neps.Categorical(   # categorical
            ("adam", "sgd", "rmsprop")
        )
        learning_rate = neps.Float(     # log spaced float
            min_value=1e-5, max_value=1, log=True
        )
        epochs = neps.Fidelity(         # fidelity integer
            neps.Integer(1, 100)
        )
        batch_size = neps.Integer(      # integer with a prior
            min_value=32,
            max_value=512,
            prior=128,
            prior_confidence="medium"
        )

    neps.run(
        evaluate_pipeline=evaluate_pipeline,
        pipeline_space=MySpace(),
        root_directory="usage_example",
        evaluations_to_spend=5,
        max_evaluations_per_run=10,
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

        pipeline_space: The search space to minimize over.

            This most direct way to specify the search space is as follows:

            ```python
            MySpace(PipelineSpace):
                dataset = "mnist"               # constant
                nlayers = neps.Integer(2,10)    # integer
                alpha = neps.Float(0.1, 1.0)    # float
                optimizer = neps.Categorical(   # categorical
                    ("adam", "sgd", "rmsprop")
                )
                learning_rate = neps.Float(     # log spaced float
                    min_value=1e-5, max_value=1, log=True
                )
                epochs = neps.Fidelity(         # fidelity integer
                    neps.Integer(1, 100)
                )
                batch_size = neps.Integer(      # integer with a prior
                    min_value=32,
                    max_value=512,
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

        write_summary_to_disk: If True, creates a csv and txt files after each worker is done,
            holding summary information about the configs and results.

        max_evaluations_per_run: Number of evaluations this specific call should do.

            ??? note "Limitation on Async mode"
                Currently, there is no specific number to control number of parallel evaluations running with
                the same worker, so in case you want to limit the number of parallel evaluations,
                it's crucial to limit the number of evaluations per run.

        evaluations_to_spend: Number of evaluations after which to terminate.
            This is shared between all workers operating in the same `root_directory`.

        continue_until_max_evaluation_completed:
            If true, only stop after evaluations_to_spend have been completed.
            This is only relevant in the parallel setting.

        cost_to_spend: No new evaluations will start when this cost is exceeded. Requires
            returning a cost in the evaluate_pipeline function, e.g.,
            `return dict(loss=loss, cost=cost)`.

        fidelities_to_spend: Number of evaluations in case of multi-fidelity after which to terminate.

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
            choose the optimizer based on the search space given.

            ??? note "Available optimizers"

                See the [optimizers documentation](../../reference/search_algorithms/landing_page_algo.md) for a list of available optimizers.

            With any optimizer choice, you also may provide some additional parameters to the optimizers.
            We do not recommend this unless you are familiar with the optimizer you are using. You
            may also specify an optimizer as a dictionary for supporting reading in serialized yaml
            formats:

            ```python
            neps.run(
                ...,
                optimzier=("priorband",
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

        warmstart_configs: A list of configurations to warmstart the NePS state with.
            This is useful for testing and debugging purposes, where you want to
            start with a set of predefined configurations and their results.
            Each configuration is a tuple of three elements:
            1. A dictionary of the samplings to make, i.e. resolution_context.samplings_made
            2. A dictionary of the environment values, i.e. resolution_context.environment_values
            3. The result of the evaluation, which is the return value of the `evaluate_pipeline`
               function, i.e. the objective value to minimize or a dictionary with
               `"objective_to_minimize"` and `"cost"` keys.

            !!! warning "Warmstarting compatibility"

                The warmstarting feature is only compatible with the new NEPS optimizers,
                such as `neps.algorithms.neps_random_search`, `neps.algorithms.neps_priorband`,
                and `neps.algorithms.complex_random_search`.

    """  # noqa: E501
    if (
        evaluations_to_spend is None
        and max_evaluations_total is None
        and max_evaluations_per_run is None
        and cost_to_spend is None
        and max_cost_total is None
        and fidelities_to_spend is None
    ):
        warnings.warn(
            "None of the following were set, this will run idefinitely until the worker"
            " process is stopped."
            f"\n * {evaluations_to_spend=}"
            f"\n * {max_evaluations_per_run=}"
            f"\n * {cost_to_spend=}"
            f"\n * {fidelities_to_spend}",
            UserWarning,
            stacklevel=2,
        )

    if max_evaluations_total is not None:
        warnings.warn(
            "`max_evaluations_total` is deprecated and will be removed in"
            " a future release. Please use `evaluations_to_spend` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        evaluations_to_spend = max_evaluations_total

    if max_cost_total is not None:
        warnings.warn(
            "`max_cost_total` is deprecated and will be removed in a future release. "
            "Please use `cost_to_spend` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        cost_to_spend = max_cost_total

    criteria = {
        "evaluations_to_spend": evaluations_to_spend,
        "max_evaluations_per_run": max_evaluations_per_run,
        "cost_to_spend": cost_to_spend,
        "fidelities_to_spend": fidelities_to_spend,
    }
    set_criteria = [k for k, v in criteria.items() if v is not None]
    if len(set_criteria) > 1:
        raise ValueError(
            f"Multiple stopping criteria specified: {', '.join(set_criteria)}. "
            "Only one is allowed."
        )

    if warmstart_configs:
        warmstart_neps(
            root_directory=Path(root_directory),
            pipeline_space=pipeline_space,
            warmstart_configs=warmstart_configs,
            optimizer=optimizer,
            overwrite_root_directory=overwrite_root_directory,
            inside_neps=True,
        )
        overwrite_root_directory = False

    logger.info(f"Starting neps.run using root directory {root_directory}")

    # Check if the pipeline_space only contains basic HPO parameters.
    # If yes, we convert it to a classic SearchSpace, to use with the old optimizers.
    # If no, we use adjust_evaluation_pipeline_for_neps_space to convert the
    # pipeline_space and only use the new NEPS optimizers.

    # If the optimizer is not a NEPS algorithm, we try to convert the pipeline_space

    neps_classic_space_compatibility = check_neps_space_compatibility(optimizer)
    if (
        neps_classic_space_compatibility in ["both", "classic"]
        and isinstance(pipeline_space, PipelineSpace)
        and not warmstart_configs
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
            "Please use one that is, such as 'random_search', "
            "'priorband', or 'complex_random_search'."
        )

    if isinstance(space, PipelineSpace):
        assert not isinstance(evaluate_pipeline, str)
        evaluate_pipeline = adjust_evaluation_pipeline_for_neps_space(
            evaluate_pipeline, space
        )

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
    }

    is_multi_fidelity = _optimizer_info["name"] in multi_fidelity_optimizers

    if is_multi_fidelity:
        if evaluations_to_spend is not None:
            raise ValueError(
                "`evaluations_to_spend` is not allowed for multi-fidelity optimizers. "
                "Only `fidelities_to_spend` or `cost_to_spend`"
            )
    elif fidelities_to_spend is not None:
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

    _launch_runtime(
        evaluation_fn=_eval,  # type: ignore
        optimizer=_optimizer_ask,
        optimizer_info=_optimizer_info,
        cost_to_spend=cost_to_spend,
        fidelities_to_spend=fidelities_to_spend,
        optimization_dir=Path(root_directory),
        evaluations_to_spend=evaluations_to_spend,
        max_evaluations_for_worker=max_evaluations_per_run,
        continue_until_max_evaluation_completed=continue_until_max_evaluation_completed,
        objective_value_on_error=objective_value_on_error,
        cost_value_on_error=cost_value_on_error,
        ignore_errors=ignore_errors,
        overwrite_optimization_dir=overwrite_root_directory,
        sample_batch_size=sample_batch_size,
        write_summary_to_disk=write_summary_to_disk,
        worker_id=worker_id,
    )

    post_run_csv(root_directory)
    root_directory = Path(root_directory)
    summary_dir = root_directory / "summary"
    if not write_summary_to_disk:
        trajectory_of_improvements(root_directory)
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
    *,
    post_run_summary: bool = True,
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
        post_run_summary (bool, optional): If True, creates a CSV file after
            trial completion, holding summary info about configs and results.

    """
    _save_results(
        user_result=user_result,
        trial_id=pipeline_id,
        root_directory=root_directory,
    )

    if post_run_summary:
        full_frame_path, short_path = post_run_csv(root_directory)
        logger.info(
            "The post run summary has been created, which is a csv file with the "
            "output of all data in the run."
            f"\nYou can find a full dataframe at: {full_frame_path}."
            f"\nYou can find a quick summary at: {short_path}."
        )


def warmstart_neps(
    pipeline_space: PipelineSpace,
    root_directory: Path | str,
    warmstart_configs: Sequence[
        tuple[
            dict[str, Any] | Mapping[str, Any],
            dict[str, Any] | Mapping[str, Any],
            EvaluatePipelineReturn,
        ]
    ],
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
    inside_neps: bool = False,  # noqa: FBT001, FBT002
) -> None:
    """Warmstart the NePS state with given configurations.
    This is useful for testing and debugging purposes, where you want to
    start with a set of predefined configurations and their results.

    Args:
        pipeline_space: The pipeline space to use for the warmstart.
        root_directory: The path to the NePS state directory.
        warmstart_configs: A list of tuples, where each tuple contains a configuration,
            environment values, and the result of the evaluation.
            The configuration is a dictionary of parameter values, the environment values
            are also a dictionary, and the result is the evaluation result.
        overwrite_root_directory: If True, the working directory will be deleted before
            starting the warmstart.

            !!! warning "Repeated warmstarting"

                When not overwriting the working directory, starting multiple NePS
                instances will result in an error. Instead, use warmstart_neps once
                on its own and then start the NePS instances.

        optimizer: The optimizer to use for the warmstart. This can be a string, a
            callable, or a tuple of a callable and a dictionary of parameters.
            If "auto", the optimizer will be chosen based on the pipeline space.

            !!! warning "Warmstarting compatibility"

                The warmstarting feature is only compatible with the new NEPS optimizers,
                such as `neps.algorithms.neps_random_search`,
                `neps.algorithms.neps_priorband`, and
                `neps.algorithms.complex_random_search`.
        inside_neps: If True, the function is called from within the NEPS runtime.
            This is used to avoid checking the compatibility of the optimizer with the
            warmstarting feature, as this is already done in the NEPS runtime.
            If False, the function will check if the optimizer is compatible with the
            warmstarting feature and raise an error if it is not.

    Raises:
        ValueError: If the optimizer is not compatible with the warmstarting feature.
        ValueError: If the warmstart config already exists in the root directory.
    """
    if not inside_neps and check_neps_space_compatibility(optimizer) != "neps":
        raise ValueError(
            "The provided optimizer is not compatible with the warmstarting feature. "
            "Please use one that is, such as 'neps_random_search', 'neps_priorband', "
            "or 'complex_random_search'."
        )
    logger.info(
        "Warmstarting neps.run with the provided"
        f" {len(warmstart_configs)} configurations using root directory"
        f" {root_directory}"
    )
    root_directory = Path(root_directory)
    if overwrite_root_directory and root_directory.is_dir():
        shutil.rmtree(root_directory)
    optimizer_ask, optimizer_info = neps.optimizers.load_optimizer(
        optimizer, pipeline_space
    )
    state = NePSState.create_or_load(
        root_directory,
        optimizer_info=optimizer_info,
        optimizer_state=OptimizationState(
            budget=None, seed_snapshot=SeedSnapshot.new_capture(), shared_state={}
        ),
    )
    for n_config, (config, env, result) in enumerate(warmstart_configs):
        try:
            _, resolution_context = neps_space.resolve(
                pipeline=pipeline_space,
                domain_sampler=neps_space.OnlyPredefinedValuesSampler(
                    predefined_samplings=config
                ),
                environment_values=env,
            )
        except ValueError as e:
            logger.error(
                "Failed to resolve the pipeline space with the provided config:"
                f" {config} and env: {env}.",
            )
            raise e

        ask_tell = AskAndTell(optimizer=optimizer_ask, worker_id="warmstart_worker")
        if pipeline_space.fidelity_attrs and isinstance(
            optimizer_ask,
            neps.optimizers.neps_bracket_optimizer._NePSBracketOptimizer,
        ):
            rung_to_fid = optimizer_ask.rung_to_fid
            fid_to_rung = {
                v: max(k for k, val in rung_to_fid.items() if val == v)
                for v in rung_to_fid.values()
            }
            fidelity_value = env[next(iter(pipeline_space.fidelity_attrs.keys()))]
            highest_rung = max(
                [
                    fid_to_rung[small_key]
                    for small_key in [key for key in fid_to_rung if key <= fidelity_value]
                ]
            )
            for rung in range(highest_rung + 1):
                # Store the config for each rung
                config_path = f"{n_config}_rung_{rung}"

                # Check if result is a UserResultDict by checking its structure
                if isinstance(result, dict) and "cost" in result:
                    # This is a UserResultDict-like dictionary
                    rung_result = result.copy()
                    rung_result["cost"] = rung_result.get("cost", 0) / (highest_rung + 1)  # type: ignore
                else:
                    # This is a simple numeric result
                    rung_result = result  # type: ignore
                trial = ask_tell.tell_custom(
                    config_id=config_path,
                    config=config,
                    result=rung_result,
                    time_sampled=time.time(),
                    time_started=time.time(),
                    time_end=time.time(),
                    previous_trial_id=f"{n_config}_rung_{rung - 1}" if rung > 0 else None,
                    location=root_directory / "configs" / config_path,
                    worker_id=f"worker_1-{socket.gethostname()}-{os.getpid()}",
                )
                trial.config = NepsCompatConverter.to_neps_config(resolution_context)
                if (root_directory / config_path).is_dir():
                    raise ValueError(
                        f"Warmstart config {n_config} already exists in"
                        f" {root_directory}. Please remove it before running the"
                        " script again."
                    )
                TrialRepo(root_directory / "configs").store_new_trial(trial)
                assert trial.report
                assert trial.metadata.evaluating_worker_id
                state.lock_and_report_trial_evaluation(
                    trial=trial,
                    report=trial.report,
                    worker_id=trial.metadata.evaluating_worker_id,
                )
                logger.info(
                    f"Warmstarted config {config_path} with result: {rung_result}."
                )

        else:
            config_path = f"{n_config}"
            trial = ask_tell.tell_custom(
                config_id=config_path,
                config=config,
                result=result,
                time_sampled=time.time(),
                time_started=time.time(),
                time_end=time.time(),
                location=root_directory / "configs" / config_path,
                worker_id=f"worker_1-{socket.gethostname()}-{os.getpid()}",
            )
            trial.config = NepsCompatConverter.to_neps_config(resolution_context)
            if (root_directory / config_path).is_dir():
                raise ValueError(
                    f"Warmstart config {n_config} already exists in {root_directory}."
                    " Please remove it before running the script again."
                )
            TrialRepo(root_directory / "configs").store_new_trial(trial)
            assert trial.report
            assert trial.metadata.evaluating_worker_id
            state.lock_and_report_trial_evaluation(
                trial=trial,
                report=trial.report,
                worker_id=trial.metadata.evaluating_worker_id,
            )
            logger.info(f"Warmstarted config {config_path} with result: {result}.")


__all__ = ["run", "save_pipeline_results", "warmstart_neps"]

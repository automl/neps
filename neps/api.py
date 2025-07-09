"""API for the neps package."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Concatenate, Literal

import neps
import neps.optimizers.algorithms
from neps.optimizers import AskFunction, OptimizerChoice, load_optimizer
from neps.runtime import _launch_runtime
from neps.space.neps_spaces.neps_space import (
    adjust_evaluation_pipeline_for_neps_space,
    convert_neps_to_classic_search_space,
)
from neps.space.neps_spaces.parameters import Pipeline
from neps.space.parsing import convert_to_space
from neps.status.status import post_run_csv
from neps.utils.common import dynamic_load_object

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace

    from neps.optimizers.algorithms import CustomOptimizer
    from neps.space import SearchSpace
    from neps.state import EvaluatePipelineReturn

logger = logging.getLogger(__name__)


def run(  # noqa: PLR0913, C901
    evaluate_pipeline: Callable[..., EvaluatePipelineReturn] | str,
    pipeline_space: ConfigurationSpace | Pipeline,
    *,
    root_directory: str | Path = "neps_results",
    overwrite_working_directory: bool = False,
    post_run_summary: bool = True,
    max_evaluations_total: int | None = None,
    max_evaluations_per_run: int | None = None,
    continue_until_max_evaluation_completed: bool = False,
    max_cost_total: int | float | None = None,
    ignore_errors: bool = False,
    objective_value_on_error: float | None = None,
    cost_value_on_error: float | None = None,
    sample_batch_size: int | None = None,
    optimizer: (
        OptimizerChoice
        | Mapping[str, Any]
        | tuple[OptimizerChoice, Mapping[str, Any]]
        | Callable[Concatenate[SearchSpace, ...], AskFunction]  # Hack, while we transit
        | Callable[Concatenate[Pipeline, ...], AskFunction]  # from SearchSpace to
        | Callable[Concatenate[SearchSpace | Pipeline, ...], AskFunction]  # Pipeline
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

    PipelineSpace(Pipeline):
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
        pipeline_space=PipelineSpace(),
        root_directory="usage_example",
        max_evaluations_total=5,
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
            PipelineSpace(Pipeline):
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
                pipeline_space=PipelineSpace()
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

        overwrite_working_directory: If true, delete the working directory at the start of
            the run. This is, e.g., useful when debugging a evaluate_pipeline function.

        post_run_summary: If True, creates a csv file after each worker is done,
            holding summary information about the configs and results.

        max_evaluations_per_run: Number of evaluations this specific call should do.

        max_evaluations_total: Number of evaluations after which to terminate.
            This is shared between all workers operating in the same `root_directory`.

        continue_until_max_evaluation_completed:
            If true, only stop after max_evaluations_total have been completed.
            This is only relevant in the parallel setting.

        max_cost_total: No new evaluations will start when this cost is exceeded. Requires
            returning a cost in the evaluate_pipeline function, e.g.,
            `return dict(loss=loss, cost=cost)`.
        ignore_errors: Ignore hyperparameter settings that threw an error and do not raise
            an error. Error configs still count towards max_evaluations_total.
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
                optimzier={
                    "name": "priorband",
                    "sample_prior_first": True,
                }
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
    if (
        max_evaluations_total is None
        and max_evaluations_per_run is None
        and max_cost_total is None
    ):
        warnings.warn(
            "None of the following were set, this will run idefinitely until the worker"
            " process is stopped."
            f"\n * {max_evaluations_total=}"
            f"\n * {max_evaluations_per_run=}"
            f"\n * {max_cost_total=}",
            UserWarning,
            stacklevel=2,
        )

    logger.info(f"Starting neps.run using root directory {root_directory}")

    # Check if the pipeline_space only contains basic HPO parameters.
    # If yes, we convert it to a classic SearchSpace, to use with the old optimizers.
    # If no, we use adjust_evaluation_pipeline_for_neps_space to convert the
    # pipeline_space and only use the new NEPS optimizers.

    # If the optimizer is not a NEPS algorithm, we try to convert the pipeline_space
    inner_optimizer = None
    if isinstance(optimizer, partial):
        inner_optimizer = optimizer.func
        while isinstance(inner_optimizer, partial):
            inner_optimizer = inner_optimizer.func
    if (
        optimizer
        not in (
            neps.optimizers.algorithms.neps_random_search,
            neps.optimizers.algorithms.neps_priorband,
            neps.optimizers.algorithms.neps_complex_random_search,
        )
        and (
            not inner_optimizer
            or inner_optimizer
            not in (
                neps.optimizers.algorithms.neps_random_search,
                neps.optimizers.algorithms.neps_priorband,
                neps.optimizers.algorithms.neps_complex_random_search,
            )
        )
        and optimizer != "auto"
    ):
        converted_space = convert_neps_to_classic_search_space(pipeline_space)
        if converted_space:
            logger.info(
                "The provided pipeline_space only contains basic HPO parameters, "
                "converting it to a classic SearchSpace."
            )
            pipeline_space = converted_space

    if isinstance(pipeline_space, Pipeline):
        assert not isinstance(evaluate_pipeline, str)
        evaluate_pipeline = adjust_evaluation_pipeline_for_neps_space(
            evaluate_pipeline, pipeline_space
        )

    space = convert_to_space(pipeline_space)
    _optimizer_ask, _optimizer_info = load_optimizer(optimizer=optimizer, space=space)

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
        max_cost_total=max_cost_total,
        optimization_dir=Path(root_directory),
        max_evaluations_total=max_evaluations_total,
        max_evaluations_for_worker=max_evaluations_per_run,
        continue_until_max_evaluation_completed=continue_until_max_evaluation_completed,
        objective_value_on_error=objective_value_on_error,
        cost_value_on_error=cost_value_on_error,
        ignore_errors=ignore_errors,
        overwrite_optimization_dir=overwrite_working_directory,
        sample_batch_size=sample_batch_size,
    )

    if post_run_summary:
        full_frame_path, short_path = post_run_csv(root_directory)
        logger.info(
            "The post run summary has been created, which is a csv file with the "
            "output of all data in the run."
            f"\nYou can find a full dataframe at: {full_frame_path}."
            f"\nYou can find a quick summary at: {short_path}."
        )
    else:
        logger.info(
            "Skipping the creation of the post run summary, which is a csv file with the "
            " output of all data in the run."
            "\nSet `post_run_summary=True` to enable it."
        )


__all__ = ["run"]

"""API for the neps package."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Concatenate, Literal

from neps.optimizers import AskFunction, OptimizerChoice, load_optimizer
from neps.runtime import _launch_runtime
from neps.space.parsing import convert_to_space
from neps.status.status import post_run_csv, trajectory_of_improvements
from neps.utils.common import dynamic_load_object

if TYPE_CHECKING:
    from ConfigSpace import ConfigurationSpace

    from neps.optimizers.algorithms import CustomOptimizer
    from neps.space import Parameter, SearchSpace
    from neps.state import EvaluatePipelineReturn

logger = logging.getLogger(__name__)


def run(  # noqa: C901, D417, PLR0913
    evaluate_pipeline: Callable[..., EvaluatePipelineReturn] | str,
    pipeline_space: (
        Mapping[str, dict | str | int | float | Parameter]
        | SearchSpace
        | ConfigurationSpace
    ),
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
        | Callable[Concatenate[SearchSpace, ...], AskFunction]
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

    pipeline_space = dict(some_parameter=neps.Float(lower=0, upper=1))
    neps.run(
        evaluate_pipeline=evaluate_pipeline,
        pipeline_space={
            "some_parameter": (0.0, 1.0),   # float
            "another_parameter": (0, 10),   # integer
            "optimizer": ["sgd", "adam"],   # categorical
            "epoch": neps.Integer(          # fidelity integer
                lower=1,
                upper=100,
                is_fidelity=True
            ),
            "learning_rate": neps.Float(    # log spaced float
                lower=1e-5,
                uperr=1,
                log=True
            ),
            "alpha": neps.Float(            # float with a prior
                lower=0.1,
                upper=1.0,
                prior=0.99,
                prior_confidence="high",
            )
        },
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

        pipeline_space: The search space to minimize over.

            This most direct way to specify the search space is as follows:

            ```python
            neps.run(
                pipeline_space={
                    "dataset": "mnist",             # constant
                    "nlayers": (2, 10),             # integer
                    "alpha": (0.1, 1.0),            # float
                    "optimizer": [                  # categorical
                        "adam", "sgd", "rmsprop"
                    ],
                    "learning_rate": neps.Float(,   # log spaced float
                        lower=1e-5, upper=1, log=True
                    ),
                    "epochs": neps.Integer(         # fidelity integer
                        lower=1, upper=100, is_fidelity=True
                    ),
                    "batch_size": neps.Integer(     # integer with a prior
                        lower=32, upper=512, prior=128
                    ),

                }
            )
            ```

            You can also directly instantiate any of the parameters
            defined by [`Parameter`][neps.space.parameters.Parameter]
            and provide them directly.

            Some important properties you can set on parameters are:

            * `prior=`: If you have a good idea about what a good setting
                for a parameter may be, you can set this as the prior for
                a parameter. You can specify this along with `prior_confidence`
                if you would like to assign a `"low"`, `"medium"`, or `"high"`
                confidence to the prior.


            !!! note "Yaml support"

                To support spaces defined in yaml, you may also define the parameters
                as dictionarys, e.g.,

                ```python
                neps.run(
                    pipeline_space={
                        "dataset": "mnist",
                        "nlayers": {"type": "int", "lower": 2, "upper": 10},
                        "alpha": {"type": "float", "lower": 0.1, "upper": 1.0},
                        "optimizer": {"type": "cat", "choices": ["adam", "sgd", "rmsprop"]},
                        "learning_rate": {"type": "float", "lower": 1e-5, "upper": 1, "log": True},
                        "epochs": {"type": "int", "lower": 1, "upper": 100, "is_fidelity": True},
                        "batch_size": {"type": "int", "lower": 32, "upper": 512, "prior": 128},
                    }
                )
                ```

            !!! note "ConfigSpace support"

                You may also use a `ConfigurationSpace` object from the
                `ConfigSpace` library.

        root_directory: The directory to save progress to.

        overwrite_root_directory: If true, delete the working directory at the start of
            the run. This is, e.g., useful when debugging a evaluate_pipeline function.

        write_summary_to_disk: If True, creates a csv and txt files after each worker is done,
            holding summary information about the configs and results.

        max_evaluations_per_run: Number of evaluations this specific call should do.

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
            If not provided, a `worker_id` will be automatically generated which follows this pattern:
            `worker_<N>` where `<N>` is a unique integer for each worker and increnements with each new worker.
            List of all workers that have been created so far is stored in
            `root_directory/optimizer_state.pkl` in the attribute `worker_ids`.

            ??? tip "Why specify a `worker_id`?"
                This is useful when you want to keep track of which worker did what in the
                results, e.g., when debugging or running on a cluster.

            ??? warning "Douplication of `worker_id`"
                Make sure that each worker has a unique `worker_id`, in case of duplication,
                for protecting against overwriting results of other workers, the optimization
                will be stopped with an error.

        optimizer: Which optimizer to use.

            Not sure which to use? Leave this at `"auto"` and neps will
            choose the optimizer based on the search space given.

            ??? note "Available optimizers"

                ---

                * `#!python "bayesian_optimization"`,

                    ::: neps.optimizers.algorithms.bayesian_optimization
                        options:
                            show_root_heading: false
                            show_signature: false
                            show_source: false

                ---

                * `#!python "ifbo"`

                    ::: neps.optimizers.algorithms.ifbo
                        options:
                            show_root_heading: false
                            show_signature: false
                            show_source: false

                ---

                * `#!python "successive_halving"`:

                    ::: neps.optimizers.algorithms.successive_halving
                        options:
                            show_root_heading: false
                            show_signature: false
                            show_source: false

                ---

                * `#!python "hyperband"`:

                    ::: neps.optimizers.algorithms.hyperband
                        options:
                            show_root_heading: false
                            show_signature: false
                            show_source: false

                ---

                * `#!python "priorband"`:

                    ::: neps.optimizers.algorithms.priorband
                        options:
                            show_root_heading: false
                            show_signature: false
                            show_source: false

                ---

                * `#!python "asha"`:

                    ::: neps.optimizers.algorithms.asha
                        options:
                            show_root_heading: false
                            show_signature: false
                            show_source: false

                ---

                * `#!python "async_hb"`:

                    ::: neps.optimizers.algorithms.async_hb
                        options:
                            show_root_heading: false
                            show_signature: false
                            show_source: false

                ---

                * `#!python "random_search"`:

                    ::: neps.optimizers.algorithms.random_search
                        options:
                            show_root_heading: false
                            show_signature: false
                            show_source: false

                ---

                * `#!python "grid_search"`:

                    ::: neps.optimizers.algorithms.grid_search
                        options:
                            show_root_heading: false
                            show_signature: false
                            show_source: false

                ---


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

    logger.info(f"Starting neps.run using root directory {root_directory}")
    space = convert_to_space(pipeline_space)
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


__all__ = ["run"]

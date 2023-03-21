from __future__ import annotations

import json
import logging
from pathlib import Path

import jahs_bench
import numpy as np
from joint_config_space import joint_config_space
from scipy.stats import kstest
from typing_extensions import Literal

import neps
from neps.optimizers.base_optimizer import BaseOptimizer
from neps.search_spaces.search_space import pipeline_space_from_configspace

OPTIMIZERS = [
    "random_search",
    "mf_bayesian_optimization",
    "bayesian_optimization",
    "regularized_evolution",
]
TASKS = ["cifar10", "fashion_mnist"]
LOSS_FILE = Path(__file__, "losses.json")

logging.basicConfig(level=logging.INFO)


def incumbent_at(root_directory: str | Path, step: int):
    """
    Return the incumbent of the run at step n

    Args:
        root_directory: root directory of the optimization run
        step: step n at which to return the incumbent
    """
    log_file = Path(root_directory, "all_losses_and_configs.txt")
    losses = [
        float(line[6:])
        for line in log_file.read_text(encoding="utf-8").splitlines()
        if "Loss: " in line
    ]
    incumbent_at_n = min(losses[:step])
    return incumbent_at_n


class RegressionRunner:
    """Setting standard configurations, running the optimizations,
    and running regression test"""

    def evaluation_func(self):
        """
        If the optimizer is cost aware, return the evaluation function with cost
        """

        self.benchmark = jahs_bench.Benchmark(
            task=self.task, kind="surrogate", download=True
        )

        def cost_evaluation(
            pipeline_directory, previous_pipeline_directory, **joint_configuration
        ):  # pylint: disable=unused-argument
            epoch = joint_configuration.pop("epoch")
            joint_configuration.update({"N": 5, "W": 16, "Resolution": 1.0})

            results = self.benchmark(joint_configuration, nepochs=epoch)
            return {
                "loss": 100 - results[epoch]["valid-acc"],
                "cost": results[epoch]["runtime"],
            }

        def loss_evaluation(
            pipeline_directory, previous_pipeline_directory, **joint_configuration
        ):  # pylint: disable=unused-argument
            epoch = joint_configuration.pop("epoch")
            joint_configuration.update({"N": 5, "W": 16, "Resolution": 1.0})

            results = self.benchmark(joint_configuration, nepochs=epoch)
            return 100 - results[epoch]["valid-acc"]

        if "cost" in self.optimizer:
            return cost_evaluation
        else:
            return loss_evaluation

    def __init__(
        self,
        optimizer: Literal[
            "default",
            "bayesian_optimization",
            "random_search",
            "cost_cooling",
            "mf_bayesian_optimization",
            "grid_search",
            "cost_cooling_bayesian_optimization",  # not implemented yet?
            "regularized_evolution",
        ]
        | BaseOptimizer
        | str = "mf_bayesian_optimization",
        iterations: int = 100,
        task: Literal["cifar10", "colorectal_histology", "fashion_mnist"]
        | str = "cifar10",
        max_evaluations: int = 150,
        budget: int = 10000,
        experiment_name: str = "",
    ):
        """
        Download benchmark, initialize Pipeline space, evaluation function and set paths,

        Args:
            optimizer: Choose an optimizer to run, this will also be the name of the run
            iterations: For how many repetitions to run the optimizations
            task: the dataset name for jahs_bench
            max_evaluations: maximum number of total evaluations
            budget: budget for cost aware methods
            experiment_name: string to identify different experiments
        """

        self.task = task
        self.optimizer = optimizer
        if experiment_name:
            experiment_name += "_"
        self.name = f"{optimizer}_{task}_{experiment_name}runs"
        self.iterations = iterations
        self.benchmark = None
        self.run_pipeline = None
        # TODO: convert string paths to Path objects
        self.root_directory = f"./{self.name}"

        # Cost cooling optimizer expects budget but none of the others does
        self.budget = budget if "cost" in optimizer else None
        self.max_evaluations = max_evaluations

        if optimizer not in OPTIMIZERS:
            ValueError(
                f"Regression hasn't been run for {optimizer} optimizer, "
                f"please update the SEARCHERS first"
            )

        self.final_losses: list[float] = []
        file_name = f"final_losses_{self.max_evaluations}_.txt"
        self.final_losses_path = Path(self.root_directory, file_name)
        if not self.final_losses_path.parent.exists():
            Path(self.root_directory).mkdir()

        self.pipeline_space = pipeline_space_from_configspace(joint_config_space)

        # Sample size for tests
        self.sample_size = 10

        # For Regularized evolution sampler ignores fidelity hyperparameters
        # by sampling None for them
        is_fidelity = self.optimizer != "regularized_evolution"
        self.pipeline_space["epoch"] = neps.IntegerParameter(
            lower=1, upper=200, is_fidelity=is_fidelity
        )

    def save_losses(self, file_name: str | None = None):
        if file_name:
            self.final_losses_path = Path(self.root_directory, file_name)
        with self.final_losses_path.open(mode="w+", encoding="utf-8") as f:
            f.writelines([str(loss) + "\n" for loss in self.final_losses])
        logging.info(
            f"Saved the results of {len(self.final_losses)} "
            f"runs of {self.max_evaluations} "
            f"max evaluations into the file: {self.final_losses_path}"
        )

    def run_neps(self, save=False):
        """
        Run iterations number of neps runs
        """
        # Retrieve the surrogate model only if we are going to run the optimizer
        if not self.run_pipeline:
            self.run_pipeline = self.evaluation_func()

        for i in range(self.iterations):
            working_directory = Path(self.root_directory, "results/test_run_" + str(i))

            neps.run(
                run_pipeline=self.run_pipeline,
                pipeline_space=self.pipeline_space,
                searcher=self.optimizer,
                budget=self.budget,
                root_directory=working_directory,
                max_evaluations_total=self.max_evaluations,
            )

            best_error = incumbent_at(working_directory, self.max_evaluations)

            self.final_losses.append(float(best_error))

        # Try to reduce memory consumption
        del self.benchmark
        self.run_pipeline = None
        if save:
            self.save_losses()

        return np.array(self.final_losses)

    def read_results(self):
        """
        Read the results of the last run.
        Either returns results of the most recent run, or
        return the values from LOSS_DICT
        """

        if self.final_losses:
            return np.array(self.final_losses)
        elif self.final_losses_path.exists():
            # Read from final_losses_path for each regression run
            self.final_losses = [
                float(loss)
                for loss in self.final_losses_path.read_text(
                    encoding="utf-8"
                ).splitlines()[: self.iterations]
            ]
        else:
            # Read from the results of previous runs if final_losses_path is not saved
            try:
                for i in range(self.iterations):
                    working_directory = Path(
                        self.root_directory, "results/test_run_" + str(i)
                    )
                    best_error = incumbent_at(working_directory, self.max_evaluations)
                    self.final_losses.append(float(best_error))
            except FileNotFoundError as not_found:
                # Try reading from the LOSS_FILE in the worst case
                if LOSS_FILE.exists():
                    with LOSS_FILE.open(mode="r", encoding="utf-8") as f:
                        loss_dict = json.load(f)
                    self.final_losses = loss_dict[self.optimizer][self.task]
                else:
                    raise FileNotFoundError(
                        f"Results from the previous runs are not "
                        f"found, and {LOSS_FILE} does not exist"
                    ) from not_found
        return np.array(self.final_losses)

    def test(self, max_evaluations=150):
        """
        Target run for the regression test, keep all the parameters same.

        Args:
            max_evaluations: Number of evaluations after which to terminate optimization.
        """

        # Retrieve the surrogate model only if we are going to run the optimizer
        if not self.run_pipeline:
            self.run_pipeline = self.evaluation_func()

        # Sample losses of self.sample_size runs
        samples = []
        for i in range(self.sample_size):
            working_directory = Path(self.root_directory, f"results/test_run_target_{i}")
            neps.run(
                run_pipeline=self.run_pipeline,
                pipeline_space=self.pipeline_space,
                searcher=self.optimizer,
                budget=self.budget,
                root_directory=working_directory,
                max_evaluations_total=max_evaluations,
            )
            best_error = incumbent_at(working_directory, max_evaluations)
            samples.append(best_error)

        # Try to reduce memory consumption
        del self.benchmark
        self.run_pipeline = None

        # Run tests
        target = self.read_results()

        threshold = self.median_threshold(target)

        ks_result = kstest(samples, target)
        median_dist = np.median(samples) - np.median(target)
        ks_test = 0 if ks_result.pvalue < 0.1 else 1
        median_test = 0 if abs(median_dist) > threshold else 1
        median_improvement = 1 if median_dist < 0 else 0

        return ks_test, median_test, median_improvement

    @staticmethod
    def median_threshold(
        target: np.ndarray, percentile: float | int = 92.5, sample_size: int = 10
    ):
        stat_size = 1000
        p_index = int(stat_size * percentile / 100)
        distances = np.zeros(stat_size)
        for i, _ in enumerate(distances):
            _sample = np.random.choice(target, size=sample_size, replace=False)
            median_dist = np.median(_sample) - np.median(target)
            distances[i] = median_dist
        distances.sort()
        return distances[p_index]


if __name__ == "__main__":
    json_file = Path("losses.json")
    if json_file.exists():
        with json_file.open(mode="r", encoding="utf-8") as f:
            losses_dict = json.load(f)
    else:
        losses_dict = dict()

    n = 100
    max_evaluations = 150
    print(f"Optimizers the results are already recorded for: {losses_dict.keys()}")
    for optimizer in OPTIMIZERS:
        if optimizer in losses_dict:
            print(f"For {optimizer} recorded tasks are: {losses_dict[optimizer].keys()}")
        for task in TASKS:
            if (
                isinstance(losses_dict.get(optimizer, None), dict)
                and len(losses_dict[optimizer].get(task, [])) == n
            ):
                continue
            else:
                runner = RegressionRunner(optimizer, n, task, max_evaluations=150)
                # runner.pipeline_space = can be customized here...
                # runner.run_pipeline = can be customized here...
                runner.run_neps(save=True)
                best_results = runner.read_results().tolist()
                minv, maxv = min(best_results), max(best_results)
                print(
                    f"For optimizer {optimizer} on {task}:\n "
                    f"\tMin of best results: {minv}\n\tMax of best results: {maxv}"
                )
                if isinstance(losses_dict.get(optimizer, None), dict) and isinstance(
                    losses_dict[optimizer].get(task, None), list
                ):
                    losses_dict[optimizer][task] = best_results
                elif isinstance(losses_dict.get(optimizer, None), dict):
                    update_dict = {task: best_results}
                    losses_dict[optimizer].update(update_dict)
                else:
                    update_dict = {optimizer: {task: best_results}}
                    losses_dict.update(update_dict)

    # print(losses_dict)
    with json_file.open(mode="w", encoding="utf-8") as f:
        json.dump(losses_dict, f)
